"""Génère un mini-dataset SYNTHÉTIQUE au format MVTec AD (plaques métalliques brossées).

    python scripts/make_synthetic_dataset.py --out data/synthetic --n-train 60 --n-test 30

Usage : tests automatisés, démo hors-ligne, vérification du pipeline de bout en bout
sans télécharger MVTec AD. Ces images ne représentent AUCUNE pièce réelle et les
métriques obtenues dessus n'ont aucune valeur de performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

CATEGORY = "synthetic_plate"


def plate(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    base = rng.uniform(140, 175)
    noise = rng.normal(0, 1, (size, size)).astype(np.float32)
    brushed = cv2.GaussianBlur(noise, (0, 0), sigmaX=12, sigmaY=0.6) * 40  # stries horizontales
    shade = np.linspace(-8, 8, size, dtype=np.float32)[None, :] * rng.uniform(-1, 1)
    g = np.clip(base + brushed + shade + rng.normal(0, 2, (size, size)), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    for cx, cy in [(24, 24), (size - 24, 24), (24, size - 24), (size - 24, size - 24)]:  # rivets
        cv2.circle(img, (cx, cy), 9, (110, 110, 115), -1)
        cv2.circle(img, (cx, cy), 9, (90, 90, 95), 1)
    return img


def add_defect(img: np.ndarray, kind: str, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if kind == "crack":
        x, y = rng.integers(60, w - 60), rng.integers(60, h - 60)
        pts = [(x, y)]
        for _ in range(int(rng.integers(6, 12))):
            x += int(rng.integers(-6, 10))
            y += int(rng.integers(-10, 10))
            pts.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
        cv2.polylines(img, [np.array(pts)], False, (45, 45, 50), 2, cv2.LINE_AA)
        cv2.polylines(mask, [np.array(pts)], False, 255, 4)
    elif kind == "scratch":
        x1, y1 = rng.integers(40, w - 40), rng.integers(40, h - 40)
        ang = rng.uniform(0, np.pi)
        L = rng.integers(40, 90)
        x2, y2 = int(x1 + L * np.cos(ang)), int(y1 + L * np.sin(ang))
        cv2.line(img, (int(x1), int(y1)), (x2, y2), (215, 215, 220), 1, cv2.LINE_AA)
        cv2.line(mask, (int(x1), int(y1)), (x2, y2), 255, 4)
    elif kind == "hole":
        c = (int(rng.integers(50, w - 50)), int(rng.integers(50, h - 50)))
        r = int(rng.integers(4, 9))
        cv2.circle(img, c, r, (30, 30, 30), -1)
        cv2.circle(mask, c, r + 2, 255, -1)
    elif kind == "contamination":
        c = (int(rng.integers(50, w - 50)), int(rng.integers(50, h - 50)))
        axes = (int(rng.integers(8, 18)), int(rng.integers(5, 12)))
        cv2.ellipse(img, c, axes, float(rng.uniform(0, 180)), 0, 360, (60, 90, 120), -1)
        cv2.ellipse(mask, c, axes, 0, 0, 360, 255, -1)
    return mask


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n-train", type=int, default=60)
    p.add_argument("--n-test", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    root = Path(args.out) / CATEGORY

    def save(path: Path, img):
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)

    for i in range(args.n_train):
        save(root / "train" / "good" / f"{i:03d}.png", plate(rng))
    for i in range(args.n_test):
        save(root / "test" / "good" / f"{i:03d}.png", plate(rng))
    for kind in ("crack", "scratch", "hole", "contamination"):
        for i in range(max(1, args.n_test // 3)):
            img = plate(rng)
            mask = add_defect(img, kind, rng)
            save(root / "test" / kind / f"{i:03d}.png", img)
            save(root / "ground_truth" / kind / f"{i:03d}_mask.png", mask)
    print(f"-> {root}")


if __name__ == "__main__":
    main()
