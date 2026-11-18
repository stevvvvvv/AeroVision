"""Convertit les masques MVTec AD en annotations YOLO segmentation (taxonomie générique).

    python scripts/mvtec_to_yolo.py --data-root data/mvtec_ad --out data/yolo_defects

Chaque composante connexe d'un masque devient une instance (polygone) de la classe
générique correspondante (voir aerovision_ml/taxonomy.py).

Limite assumée : MVTec AD contient ~1 250 images défectueuses réparties sur 15 pièces.
C'est peu pour un détecteur supervisé : YOLO sert ici à NOMMER le type de défaut sur les
zones trouvées par PatchCore, pas à remplacer la détection non supervisée.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aerovision_ml import data  # noqa: E402
from aerovision_ml.taxonomy import GENERIC_DEFECTS, generic_index, to_generic  # noqa: E402


def mask_to_polygons(mask, min_area: int = 20):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, 0.003 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(c) < 3:
            continue
        polys.append([v for x, y in c for v in (x / w, y / h)])
    return polys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="data/yolo_defects")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--include-good", type=int, default=20, help="images saines par catégorie (négatifs)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    rng = random.Random(args.seed)
    counts = {c: 0 for c in GENERIC_DEFECTS}
    for cat in data.list_categories(args.data_root):
        samples = data.test_samples(args.data_root, cat)
        defective = [s for s in samples if s.is_anomalous and s.mask_path]
        good = [s for s in samples if not s.is_anomalous]
        rng.shuffle(good)
        for s in defective + good[: args.include_good]:
            split = "val" if rng.random() < args.val_fraction else "train"
            stem = f"{cat}_{s.defect}_{s.path.stem}"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy(s.path, out / "images" / split / f"{stem}{s.path.suffix}")
            lines = []
            if s.is_anomalous:
                img = data.load_image(s.path)
                cls = to_generic(cat, s.defect)
                for poly in mask_to_polygons(data.load_mask(s.mask_path, img.shape[:2])):
                    lines.append(f"{generic_index(cls)} " + " ".join(f"{v:.5f}" for v in poly))
                    counts[cls] += 1
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))

    yaml = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(GENERIC_DEFECTS)]
    (out / "dataset.yaml").write_text("\n".join(yaml) + "\n")
    print("Instances par classe :", counts)
    print(f"-> {out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
