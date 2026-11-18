"""Métriques standard du benchmark MVTec AD."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from .calibration import best_f1_threshold, subsample_pixels


def image_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float | None = None) -> dict:
    out = {"n": int(len(labels)), "n_defective": int(labels.sum())}
    if len(np.unique(labels)) == 2:
        out["image_auroc"] = float(roc_auc_score(labels, scores))
        _, out["image_f1_oracle"] = best_f1_threshold(scores, labels)
    if threshold is not None:
        pred = (scores >= threshold).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        out.update(
            image_f1_at_threshold=float(f1_score(labels, pred, zero_division=0)),
            recall=tp / max(1, tp + fn),            # taux de détection des défauts
            false_negative_rate=fn / max(1, tp + fn),  # LE chiffre critique en aéro
            false_positive_rate=fp / max(1, fp + tn),
            confusion={"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        )
    return out


def pixel_metrics(maps: list[np.ndarray], masks: list[np.ndarray], threshold: float | None = None
                  ) -> dict:
    s, y = subsample_pixels(maps, masks)
    out: dict = {}
    if len(np.unique(y)) == 2:
        out["pixel_auroc"] = float(roc_auc_score(y, s))
    if threshold is not None:
        pred = (s >= threshold).astype(int)
        inter = int(((pred == 1) & (y == 1)).sum())
        union = int(((pred == 1) | (y == 1)).sum())
        out["pixel_iou_sampled"] = inter / max(1, union)
    return out
