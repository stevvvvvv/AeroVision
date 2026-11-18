"""Détecteur de référence en vision classique (OpenCV, SANS apprentissage).

Rôle :
- faire tourner l'application de bout en bout avant tout entraînement ;
- servir de point de comparaison honnête : « que gagne-t-on avec le deep learning ? ».

Méthode : on estime le fond de la pièce par un flou large, puis on combine
(a) l'écart local au fond et (b) un « black-hat » morphologique qui fait ressortir les
structures fines et sombres (fissures, rayures). Fonctionne correctement sur surfaces
uniformes, mal sur pièces texturées ou complexes : c'est attendu et documenté.
"""

from __future__ import annotations

import cv2
import numpy as np

BASELINE_PIXEL_THRESHOLD = 0.5


def anomaly_map(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(gray).astype(np.float32)

    sigma_bg = max(h, w) / 20
    background = cv2.GaussianBlur(g, (0, 0), sigma_bg)
    diff = np.abs(g - background)

    k = max(7, (min(h, w) // 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)

    def robust01(x: np.ndarray) -> np.ndarray:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med))) + 1e-6
        z = (x - med) / (1.4826 * mad)
        return np.clip(z / 12.0, 0, 1)   # z = 12 -> 1.0 ; z = 6 -> 0.5 (seuil)

    combined = np.maximum(robust01(diff), robust01(blackhat))
    return cv2.GaussianBlur(combined, (0, 0), 2).astype(np.float32)


def image_score(amap: np.ndarray) -> float:
    """Score global : moyenne du top 0,1 % des pixels (plus robuste que le max seul)."""
    flat = amap.ravel()
    k = max(1, int(0.001 * flat.size))
    return float(np.partition(flat, -k)[-k:].mean())
