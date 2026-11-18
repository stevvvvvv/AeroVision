"""Calibration des seuils et du score de confiance.

Un score d'anomalie brut (distance dans l'espace des features) n'a pas de sens pour un
opérateur. On le transforme en :
- une décision binaire au seuil qui maximise le F1 sur un jeu de calibration ;
- une probabilité calibrée (régression logistique 1D, « Platt scaling ») affichée comme
  score de confiance.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Seuil qui maximise le F1. Retourne (seuil, f1)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    f1 = f1[:-1]  # le dernier point n'a pas de seuil associé
    if len(thresholds) == 0:
        return float(np.median(scores)), 0.0
    i = int(np.nanargmax(f1))
    return float(thresholds[i]), float(f1[i])


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Retourne (a, b) tels que P(défaut) = sigmoid(a * score + b)."""
    if len(np.unique(labels)) < 2:
        raise ValueError("Il faut des exemples sains ET défectueux pour calibrer.")
    lr = LogisticRegression(C=1e3, max_iter=1000)
    lr.fit(scores.reshape(-1, 1), labels)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_from_normal_only(normal_scores: np.ndarray, z_threshold: float = 3.0
                           ) -> tuple[float, float, float]:
    """Repli quand on n'a AUCUN défaut étiqueté : seuil = moyenne + z·écart-type des
    pièces saines, et une sigmoïde centrée sur ce seuil. Moins fiable — signalé dans l'UI."""
    mu, sd = float(normal_scores.mean()), float(normal_scores.std() + 1e-6)
    thr = mu + z_threshold * sd
    a = 4.0 / sd          # pente arbitraire mais documentée
    b = -a * thr
    return thr, a, b


def subsample_pixels(maps: list[np.ndarray], masks: list[np.ndarray], n: int = 200_000,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Échantillonne des pixels (scores, labels) pour calibrer/évaluer sans exploser la RAM."""
    s = np.concatenate([m.ravel() for m in maps])
    y = np.concatenate([k.ravel() for k in masks])
    if len(s) > n:
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        n_pos = min(len(pos), n // 2)
        idx = np.concatenate([rng.choice(pos, n_pos, replace=False) if n_pos else pos[:0],
                              rng.choice(neg, min(len(neg), n - n_pos), replace=False)])
        s, y = s[idx], y[idx]
    return s, y
