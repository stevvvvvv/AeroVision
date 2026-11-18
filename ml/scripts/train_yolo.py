"""Entraîne YOLO (segmentation d'instances) sur la taxonomie générique de défauts.

    pip install ultralytics
    python scripts/train_yolo.py --data data/yolo_defects/dataset.yaml --epochs 100

Le meilleur poids est copié dans models/yolo_defects.pt, que le backend charge
automatiquement s'il est présent.

Note licence : Ultralytics est sous AGPL-3.0. Pour un usage commercial fermé, il faut une
licence Ultralytics ou un détecteur sous licence permissive (ex. RT-DETR via
une implémentation Apache-2.0). Voir docs/donnees.md.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolov8n-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="models/yolo_defects.pt")
    args = p.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("pip install ultralytics") from e

    model = YOLO(args.model)
    res = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, patience=20,
                      # augmentations raisonnables pour de l'inspection : pas de mosaic agressif
                      mosaic=0.5, degrees=10, fliplr=0.5, hsv_v=0.3, project="runs", name="aerovision")
    best = Path(res.save_dir) / "weights" / "best.pt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
