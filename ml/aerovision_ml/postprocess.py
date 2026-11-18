"""Post-traitement OpenCV : carte d'anomalie -> masque -> régions localisées.

C'est ici que la sortie « brute » d'un modèle devient quelque chose d'exploitable par un
opérateur : des zones encadrées, mesurées (surface en pixels), avec un score par zone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    area_px: int
    max_score: float
    mean_score: float
    polygon: list[list[int]] = field(default_factory=list)
    defect_type: str = "unknown"
    defect_type_confidence: float | None = None
    source: str = "anomaly_map"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


def binarize(anomaly_map: np.ndarray, threshold: float, open_kernel: int = 3,
             close_kernel: int = 7) -> np.ndarray:
    """Seuillage + morphologie : supprime le bruit isolé, referme les zones fragmentées."""
    mask = (anomaly_map >= threshold).astype(np.uint8) * 255
    if open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def extract_regions(anomaly_map: np.ndarray, threshold: float, min_area: int | None = None,
                    max_regions: int = 20, source: str = "anomaly_map") -> tuple[list[Region], np.ndarray]:
    """Retourne les régions anormales triées par score décroissant et le masque binaire."""
    h, w = anomaly_map.shape[:2]
    if min_area is None:
        min_area = max(16, int(0.0005 * h * w))   # 0,05 % de l'image
    mask = binarize(anomaly_map, threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[Region] = []
    for c in contours:
        area = int(cv2.contourArea(c))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        region_mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(region_mask, [c], -1, 1, thickness=-1)
        values = anomaly_map[region_mask.astype(bool)]
        eps = 0.01 * cv2.arcLength(c, True)
        poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).tolist()
        regions.append(Region(int(x), int(y), int(bw), int(bh), area,
                              float(values.max()), float(values.mean()), poly, source=source))
    regions.sort(key=lambda r: r.max_score, reverse=True)
    return regions[:max_regions], mask


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def normalize_map(anomaly_map: np.ndarray, low: float, high: float) -> np.ndarray:
    """Ramène la carte dans [0, 1] avec des bornes FIXES (et non min/max de l'image),
    pour qu'une pièce saine reste « froide » visuellement."""
    if high <= low:
        high = low + 1e-6
    return np.clip((anomaly_map - low) / (high - low), 0.0, 1.0)


def heatmap(anomaly_map_01: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap((anomaly_map_01 * 255).astype(np.uint8), cv2.COLORMAP_JET)


def render_overlay(image_bgr: np.ndarray, anomaly_map_01: np.ndarray, regions: list[Region],
                   alpha: float = 0.45) -> np.ndarray:
    """Image + carte de chaleur + boîtes numérotées (même numérotation que l'API)."""
    over = cv2.addWeighted(image_bgr, 1 - alpha, heatmap(anomaly_map_01), alpha, 0)
    for i, r in enumerate(regions, start=1):
        cv2.rectangle(over, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 0, 255), 2)
        cv2.putText(over, str(i), (r.x, max(12, r.y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(over, str(i), (r.x, max(12, r.y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return over
