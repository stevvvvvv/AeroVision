"""Chargement d'un dataset au format MVTec AD.

Arborescence attendue :

    <root>/<category>/train/good/*.png
    <root>/<category>/test/good/*.png
    <root>/<category>/test/<defect>/*.png
    <root>/<category>/ground_truth/<defect>/<name>_mask.png

Ce format est aussi celui de VisA (après conversion officielle) et de la plupart des
datasets d'anomalie industrielle : on peut donc brancher d'autres données sans changer le code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    path: Path
    category: str
    defect: str               # "good" ou libellé MVTec
    mask_path: Path | None    # None pour les images saines

    @property
    def is_anomalous(self) -> bool:
        return self.defect != "good"


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def list_categories(root: str | Path) -> list[str]:
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if (p / "train" / "good").is_dir())


def train_samples(root: str | Path, category: str) -> list[Sample]:
    folder = Path(root) / category / "train" / "good"
    return [Sample(p, category, "good", None) for p in _images(folder)]


def test_samples(root: str | Path, category: str) -> list[Sample]:
    base = Path(root) / category
    out: list[Sample] = []
    for defect_dir in sorted((base / "test").iterdir()):
        if not defect_dir.is_dir():
            continue
        defect = defect_dir.name
        for p in _images(defect_dir):
            mask = None
            if defect != "good":
                cand = base / "ground_truth" / defect / f"{p.stem}_mask.png"
                mask = cand if cand.exists() else None
            out.append(Sample(p, category, defect, mask))
    return out


def split_calibration(samples: list[Sample], fraction: float, seed: int = 0
                      ) -> tuple[list[Sample], list[Sample]]:
    """Coupe le jeu de test en (calibration, évaluation), stratifié par type de défaut.

    MVTec AD ne fournit pas de jeu de validation contenant des défauts. Pour calibrer
    honnêtement les seuils sans tricher sur l'évaluation, on réserve une fraction du
    test à la calibration et on n'évalue QUE sur le reste.
    """
    rng = random.Random(seed)
    by_defect: dict[str, list[Sample]] = {}
    for s in samples:
        by_defect.setdefault(s.defect, []).append(s)
    calib, eval_ = [], []
    for group in by_defect.values():
        group = group[:]
        rng.shuffle(group)
        k = max(1, int(round(len(group) * fraction))) if len(group) > 1 else 0
        calib += group[:k]
        eval_ += group[k:]
    return calib, eval_


def load_image(path: str | Path) -> np.ndarray:
    """Charge une image en BGR uint8 (convention OpenCV)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_mask(path: str | Path | None, shape: tuple[int, int]) -> np.ndarray:
    """Masque binaire (H, W) en {0, 1}. Masque vide si path est None."""
    if path is None:
        return np.zeros(shape, dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)
