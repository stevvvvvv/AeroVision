"""Exporte une démo statique : vraies inférences pré-calculées pour GitHub Pages.

    python tasks.py export-demo          (ou : python scripts/export_static_demo.py)

Pour chaque catégorie entraînée, on prend des images du split d'ÉVALUATION (jamais vues
pendant l'entraînement ni la calibration) : un exemple par type de défaut + des pièces
saines. Chaque image passe dans le vrai moteur (classifieur -> PatchCore -> OpenCV), et
les résultats sont écrits dans frontend/public/demo/ :

    demo/system.json          modèles, métriques, politique de décision
    demo/inspections.json     résultats d'inspection
    demo/<id>/original.jpg | heatmap.jpg | overlay.jpg | mask.png

Les images MVTec AD redistribuées restent sous licence CC BY-NC-SA 4.0 (attribution
dans demo/LICENSE-MVTEC.txt).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))

from aerovision_ml.postprocess import Region, heatmap, render_overlay  # noqa: E402
from aerovision_ml.taxonomy import (GENERIC_LABELS_FR, GENERIC_TO_AERO_HINT,  # noqa: E402
                                    to_generic)

LICENSE_NOTE = """Images issues du dataset MVTec AD
(c) MVTec Software GmbH — https://www.mvtec.com/company/research/datasets/mvtec-ad
Licence : Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
(CC BY-NC-SA 4.0) — https://creativecommons.org/licenses/by-nc-sa/4.0/

Les images dérivées (cartes d'anomalie, superpositions, masques) sont distribuées sous la
même licence. Usage non commercial uniquement.

Référence : Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for
Unsupervised Anomaly Detection", CVPR 2019.
"""


def scale_region(r: Region, s: float) -> Region:
    return Region(
        x=int(round(r.x * s)), y=int(round(r.y * s)),
        w=max(1, int(round(r.w * s))), h=max(1, int(round(r.h * s))),
        area_px=int(round(r.area_px * s * s)), max_score=r.max_score, mean_score=r.mean_score,
        polygon=[[int(round(x * s)), int(round(y * s))] for x, y in r.polygon],
        defect_type=r.defect_type, defect_type_confidence=r.defect_type_confidence, source=r.source)


def pick_samples(model, per_defect: int, n_good: int, rng: random.Random) -> list[Path]:
    files = [Path(p) for p in model.meta.get("evaluation_files", [])]
    by_defect: dict[str, list[Path]] = {}
    for f in files:
        by_defect.setdefault(f.parent.name, []).append(f)
    out = []
    for defect, group in sorted(by_defect.items()):
        group = sorted(group)
        rng.shuffle(group)
        out += group[: (n_good if defect == "good" else per_defect)]
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default=str(ROOT / "models"))
    p.add_argument("--out", default=str(ROOT / "frontend" / "public" / "demo"))
    p.add_argument("--per-defect", type=int, default=1)
    p.add_argument("--n-good", type=int, default=2)
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from app.config import Settings
    from app.inference import InspectionEngine

    engine = InspectionEngine(Settings(models_dir=Path(args.models_dir)))
    if not engine.patchcore:
        sys.exit("Aucun modèle PatchCore dans models/patchcore : entraîner d'abord.")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rng = random.Random(args.seed)
    t0 = datetime.now(timezone.utc)
    inspections = []
    det_id = 1

    for cat, model in sorted(engine.patchcore.items()):
        samples = pick_samples(model, args.per_defect, args.n_good, rng)
        print(f"{cat}: {len(samples)} images")
        for path in samples:
            path = path if path.is_absolute() else ROOT / path
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                print(f"  illisible : {path}")
                continue
            res = engine.inspect(image)          # routage par le classifieur, comme en production

            h, w = image.shape[:2]
            s = min(1.0, args.max_side / max(h, w))
            small = cv2.resize(image, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
            amap = cv2.resize(res.anomaly_map_01, (small.shape[1], small.shape[0]))
            mask = cv2.resize(res.mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
            regions = [scale_region(r, s) for r in res.regions]

            defect = path.parent.name
            iid = f"{cat}-{defect}-{path.stem}"
            d = out / iid
            d.mkdir()
            jpg = [cv2.IMWRITE_JPEG_QUALITY, 85]
            cv2.imwrite(str(d / "original.jpg"), small, jpg)
            cv2.imwrite(str(d / "heatmap.jpg"), heatmap(amap), jpg)
            cv2.imwrite(str(d / "overlay.jpg"), render_overlay(small, amap, regions), jpg)
            cv2.imwrite(str(d / "mask.png"), mask)

            dets = []
            for k, r in enumerate(regions, 1):
                dets.append({"id": det_id, "rank": k, **{f: getattr(r, f) for f in (
                    "x", "y", "w", "h", "area_px", "max_score", "mean_score", "polygon",
                    "defect_type", "defect_type_confidence", "source")}})
                det_id += 1
            created = t0 - timedelta(minutes=7 * len(inspections))
            inspections.append({
                "id": iid,
                "created_at": created.isoformat(),
                "part_reference": f"MVTec · {cat} · {defect}/{path.name}",
                "original_filename": path.name,
                "width": small.shape[1], "height": small.shape[0],
                "category": res.category, "category_confidence": res.category_confidence,
                "category_source": res.category_source, "engine": res.engine,
                "model_version": res.model_version, "calibration_source": res.calibration_source,
                "anomaly_score": res.anomaly_score, "confidence": res.confidence,
                "verdict": res.verdict, "status": "pending_review", "final_decision": None,
                "processing_ms": res.processing_ms, "warnings": res.warnings,
                "detections": dets, "reviews": [], "n_detections": len(dets),
                "aero_hints": {r.defect_type: GENERIC_TO_AERO_HINT.get(r.defect_type, "") for r in regions},
                "ground_truth": {
                    "category": cat, "defect": defect, "is_anomalous": defect != "good",
                    "generic_type": None if defect == "good" else to_generic(cat, defect),
                },
            })
            gt = "sain" if defect == "good" else defect
            print(f"  {path.name:8s} vérité={gt:22s} modèle={res.verdict:22s} "
                  f"conf={res.confidence:.2f} zones={len(dets)} pièce={res.category}")

    info = engine.info()
    info.update(defect_labels_fr=GENERIC_LABELS_FR, demo=True,
                generated_at=t0.isoformat(),
                disclaimer="Démonstrateur d'aide à l'inspection. Non qualifié pour la libération "
                           "de pièces aéronautiques ; décision finale humaine obligatoire.")
    (out / "system.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "inspections.json").write_text(json.dumps(inspections, ensure_ascii=False), encoding="utf-8")
    (out / "LICENSE-MVTEC.txt").write_text(LICENSE_NOTE, encoding="utf-8")

    ok = sum(1 for i in inspections if i["ground_truth"]["is_anomalous"] == (i["verdict"] != "conforme_probable"))
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"\n{len(inspections)} inspections exportées ({size:.1f} Mo) -> {out}")
    print(f"Concordance verdict / vérité terrain : {ok}/{len(inspections)}")


if __name__ == "__main__":
    main()
