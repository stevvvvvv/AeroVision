"""Moteur d'inspection : orchestre classification -> anomalie -> localisation -> typage.

Chaîne :
  1. Classification de la pièce (CNN)            -> choix du modèle d'anomalie
  2. PatchCore de la catégorie                   -> score + carte d'anomalie (segmentation)
  3. Post-traitement OpenCV                      -> régions encadrées
  4. YOLO (optionnel)                            -> type de défaut sur chaque région
  5. Calibration + politique de décision         -> confiance et priorité de revue

Si aucun modèle n'est disponible, repli sur le détecteur OpenCV classique, signalé
explicitement dans le résultat (champ `engine` + avertissement).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerovision_ml import baseline
from aerovision_ml.classifier import PartClassifier
from aerovision_ml.patchcore import PatchCore
from aerovision_ml.postprocess import Region, extract_regions, iou, normalize_map
from aerovision_ml.taxonomy import GENERIC_DEFECTS

from .config import Settings

log = logging.getLogger("aerovision.inference")


@dataclass
class InspectionOutput:
    category: str
    category_confidence: float | None
    category_source: str
    engine: str
    model_version: str
    calibration_source: str
    anomaly_score: float
    confidence: float
    verdict: str
    regions: list[Region]
    anomaly_map_01: np.ndarray
    mask: np.ndarray
    processing_ms: int
    warnings: list[str] = field(default_factory=list)


class InspectionEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.patchcore: dict[str, PatchCore] = {}
        self.classifier: PartClassifier | None = None
        self.yolo = None
        self._load()

    # ------------------------------------------------------------------ chargement
    def _load(self) -> None:
        mdir = Path(self.settings.models_dir)
        for path in sorted((mdir / "patchcore").glob("*.pt")):
            try:
                m = PatchCore.load(path, device=self.settings.device)
                self.patchcore[m.category or path.stem] = m
                log.info("PatchCore chargé : %s", path.name)
            except Exception as e:  # un modèle corrompu ne doit pas bloquer les autres
                log.error("Échec chargement %s : %s", path, e)
        clf = mdir / "classifier.pt"
        if clf.exists():
            self.classifier = PartClassifier.load(clf, device=self.settings.device)
        yolo = mdir / "yolo_defects.pt"
        if yolo.exists():
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(str(yolo))
            except ImportError:
                log.warning("models/yolo_defects.pt présent mais ultralytics non installé")

    def info(self) -> dict:
        return {
            "mode": "patchcore" if self.patchcore else "baseline_opencv",
            "patchcore_categories": sorted(self.patchcore),
            "patchcore_reports": {c: m.meta.get("report") for c, m in self.patchcore.items()},
            "classifier": None if not self.classifier else {
                "classes": self.classifier.classes, **self.classifier.meta},
            "yolo": self.yolo is not None,
            "defect_classes": GENERIC_DEFECTS,
            "decision_policy": {"review_low": self.settings.review_low,
                                "review_high": self.settings.review_high},
        }

    # ------------------------------------------------------------------ décision
    def _verdict(self, confidence: float) -> str:
        if confidence >= self.settings.review_high:
            return "non_conforme_probable"
        if confidence <= self.settings.review_low:
            return "conforme_probable"
        return "a_verifier"

    def _route(self, image: np.ndarray, hint: str | None) -> tuple[str, float | None, str, list[str]]:
        warnings: list[str] = []
        if hint:
            if self.patchcore and hint not in self.patchcore:
                warnings.append(f"Aucun modèle pour la catégorie « {hint} »")
            return hint, None, "user", warnings
        if self.classifier:
            cat, conf, _ = self.classifier.predict(image)
            if conf < 0.6:
                warnings.append(f"Pièce identifiée avec une confiance faible ({conf:.0%}) — vérifier la catégorie")
            return cat, conf, "classifier", warnings
        if len(self.patchcore) == 1:
            return next(iter(self.patchcore)), None, "single_model", warnings
        return "unknown", None, "none", warnings

    def _type_regions(self, image: np.ndarray, regions: list[Region]) -> list[Region]:
        """Attribue un type de défaut aux régions via YOLO ; ajoute les détections YOLO isolées."""
        if self.yolo is None:
            return regions
        res = self.yolo.predict(image, conf=self.settings.yolo_min_conf, verbose=False)[0]
        names = res.names
        for box, cls, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy()):
            b = tuple(int(v) for v in box)
            best = max(regions, key=lambda r: iou(r.box, b), default=None)
            if best is not None and iou(best.box, b) >= self.settings.yolo_match_iou:
                if best.defect_type_confidence is None or conf > best.defect_type_confidence:
                    best.defect_type, best.defect_type_confidence = names[int(cls)], float(conf)
                    best.source = f"{best.source}+yolo"
            else:
                x1, y1, x2, y2 = b
                regions.append(Region(x1, y1, x2 - x1, y2 - y1, (x2 - x1) * (y2 - y1), float(conf),
                                      float(conf), [], names[int(cls)], float(conf), "yolo"))
        return regions

    # ------------------------------------------------------------------ inspection
    def inspect(self, image: np.ndarray, category_hint: str | None = None) -> InspectionOutput:
        t0 = time.perf_counter()
        category, cat_conf, cat_source, warnings = self._route(image, category_hint)
        model = self.patchcore.get(category)

        if model is not None:
            scores, maps = model.predict(image)
            score, amap = float(scores[0]), maps[0]
            cal = model.calibration
            confidence = cal.confidence(score)
            regions, mask = extract_regions(amap, cal.pixel_threshold, source="patchcore")
            # bornes fixes de visualisation : « normal » -> 0, 2x le seuil -> 1
            low = cal.normal_mean
            high = max(cal.pixel_threshold * 1.5, low + 1e-3)
            amap01 = normalize_map(amap, low, high)
            engine = "patchcore"
            version = f"patchcore/{model.config.backbone}/{category}"
            cal_source = cal.source
            if cal.source == "normal_only":
                warnings.append("Seuils calibrés sans défauts étiquetés : confiance indicative")
            if not model.config.pretrained:
                warnings.append("Modèle de test (backbone non pré-entraîné) : résultats sans valeur")
        else:
            amap = baseline.anomaly_map(image)
            score = baseline.image_score(amap)
            confidence = float(np.clip(score, 0, 1))
            regions, mask = extract_regions(amap, baseline.BASELINE_PIXEL_THRESHOLD, source="baseline_opencv")
            amap01 = amap
            engine, version, cal_source = "baseline_opencv", "opencv-baseline/1", "heuristic"
            warnings.append("Détecteur OpenCV classique (sans IA) : aucun modèle entraîné pour cette pièce")

        regions = self._type_regions(image, regions)
        verdict = self._verdict(confidence)
        h, w = image.shape[:2]
        if mask is not None and (mask > 0).mean() > 0.4:
            warnings.append("Plus de 40 % de l'image est anormale : image probablement hors domaine "
                            "(mauvaise pièce, cadrage ou éclairage différent de l'entraînement)")
        if regions and verdict == "conforme_probable":
            warnings.append("Zones suspectes localisées malgré un score global faible")
        if not regions and verdict != "conforme_probable":
            warnings.append("Score global élevé mais aucune zone isolée : anomalie diffuse, "
                            "consulter la carte d'anomalie")

        return InspectionOutput(
            category=category, category_confidence=cat_conf, category_source=cat_source,
            engine=engine, model_version=version, calibration_source=cal_source,
            anomaly_score=score, confidence=confidence, verdict=verdict,
            regions=regions, anomaly_map_01=amap01, mask=mask,
            processing_ms=int((time.perf_counter() - t0) * 1000), warnings=warnings,
        )
