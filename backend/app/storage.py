from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from aerovision_ml.postprocess import heatmap, render_overlay

from .config import get_settings

ARTIFACTS = ("original", "heatmap", "overlay", "mask")


def inspection_dir(inspection_id: str) -> Path:
    d = Path(get_settings().storage_dir) / inspection_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_artifacts(inspection_id: str, image: np.ndarray, amap01: np.ndarray, mask: np.ndarray,
                   regions) -> None:
    d = inspection_dir(inspection_id)
    cv2.imwrite(str(d / "original.png"), image)
    cv2.imwrite(str(d / "heatmap.png"), heatmap(amap01))
    cv2.imwrite(str(d / "overlay.png"), render_overlay(image, amap01, regions))
    cv2.imwrite(str(d / "mask.png"), mask)


def artifact_path(inspection_id: str, kind: str) -> Path:
    if kind not in ARTIFACTS:
        raise ValueError(kind)
    return Path(get_settings().storage_dir) / inspection_id / f"{kind}.png"
