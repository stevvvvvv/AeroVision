"""Entraîne un PatchCore par catégorie, calibre les seuils et évalue sur un split tenu à l'écart.

Exemples :
    python scripts/train_patchcore.py --data-root data/mvtec_ad --category metal_nut
    python scripts/train_patchcore.py --data-root data/mvtec_ad --category all --coreset-ratio 0.01

Sorties (models/patchcore/) :
    <category>.pt            modèle + calibration
    <category>_report.json   métriques sur le split d'évaluation
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aerovision_ml import data  # noqa: E402
from aerovision_ml.calibration import (best_f1_threshold, fit_platt,  # noqa: E402
                                       platt_from_normal_only, subsample_pixels)
from aerovision_ml.metrics import image_metrics, pixel_metrics  # noqa: E402
from aerovision_ml.patchcore import Calibration, PatchCore, PatchCoreConfig  # noqa: E402


def train_one(args: argparse.Namespace, category: str) -> dict:
    print(f"\n=== {category} ===")
    train = data.train_samples(args.data_root, category)
    rng = random.Random(args.seed)
    rng.shuffle(train)
    n_holdout = max(2, int(len(train) * args.normal_holdout))
    holdout, fit_set = train[:n_holdout], train[n_holdout:]

    cfg = PatchCoreConfig(backbone=args.backbone, image_size=args.image_size,
                          coreset_ratio=args.coreset_ratio, seed=args.seed)
    model = PatchCore(cfg, device=args.device, pretrained=not args.smoke_test)
    model.category = category
    model.fit([data.load_image(s.path) for s in fit_set])

    # 1) statistiques sur pièces saines jamais vues par la mémoire
    normal_scores, _ = model.predict([data.load_image(s.path) for s in holdout])

    # 2) calibration sur une partie du test, évaluation sur le reste
    tests = data.test_samples(args.data_root, category)
    calib, evaluation = data.split_calibration(tests, args.calibration_fraction, args.seed)

    def run(samples):
        imgs = [data.load_image(s.path) for s in samples]
        scores, maps = model.predict(imgs)
        masks = [data.load_mask(s.mask_path, m.shape) for s, m in zip(samples, maps)]
        labels = np.array([int(s.is_anomalous) for s in samples])
        return scores, maps, masks, labels

    c_scores, c_maps, c_masks, c_labels = run(calib)
    if len(np.unique(c_labels)) == 2:
        img_thr, _ = best_f1_threshold(c_scores, c_labels)
        a, b = fit_platt(c_scores, c_labels)
        ps, py = subsample_pixels(c_maps, c_masks, seed=args.seed)
        pix_thr, _ = best_f1_threshold(ps, py)
        source = "calibration_split"
    else:
        img_thr, a, b = platt_from_normal_only(normal_scores)
        pix_thr = img_thr
        source = "normal_only"

    model.calibration = Calibration(
        image_threshold=img_thr, pixel_threshold=pix_thr, platt_a=a, platt_b=b,
        normal_mean=float(normal_scores.mean()), normal_std=float(normal_scores.std() + 1e-6),
        source=source)

    e_scores, e_maps, e_masks, e_labels = run(evaluation)
    report = {
        "category": category,
        "backbone": args.backbone,
        "coreset_ratio": args.coreset_ratio,
        "split": {"fit": len(fit_set), "normal_holdout": len(holdout),
                  "calibration": len(calib), "evaluation": len(evaluation)},
        "calibration": model.calibration.__dict__,
        "evaluation_image": image_metrics(e_scores, e_labels, img_thr),
        "evaluation_pixel": pixel_metrics(e_maps, e_masks, pix_thr),
    }
    model.meta["report"] = {k: report[k] for k in ("evaluation_image", "evaluation_pixel", "split")}
    model.meta["evaluation_files"] = [str(s.path) for s in evaluation]

    out = Path(args.out_dir)
    model.save(out / f"{category}.pt")
    (out / f"{category}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    ei, ep = report["evaluation_image"], report["evaluation_pixel"]
    print(f"AUROC image {ei.get('image_auroc', float('nan')):.3f} | "
          f"AUROC pixel {ep.get('pixel_auroc', float('nan')):.3f} | "
          f"rappel {ei.get('recall', 0):.2%} | faux négatifs {ei.get('false_negative_rate', 0):.2%}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--category", default="all", help="catégorie ou 'all'")
    p.add_argument("--out-dir", default="models/patchcore")
    p.add_argument("--backbone", default="wide_resnet50_2")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--coreset-ratio", type=float, default=0.1)
    p.add_argument("--normal-holdout", type=float, default=0.1,
                   help="fraction des images saines d'entraînement gardées hors mémoire")
    p.add_argument("--calibration-fraction", type=float, default=0.3)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke-test", action="store_true",
                   help="backbone NON pré-entraîné (test hors-ligne du pipeline, métriques sans valeur)")
    args = p.parse_args()

    cats = data.list_categories(args.data_root) if args.category == "all" else [args.category]
    reports = [train_one(args, c) for c in cats]

    lines = ["| Catégorie | AUROC image | AUROC pixel | Rappel | Taux faux négatifs | Taux faux positifs |",
             "|---|---|---|---|---|---|"]
    for r in reports:
        ei, ep = r["evaluation_image"], r["evaluation_pixel"]
        lines.append(f"| {r['category']} | {ei.get('image_auroc', 0):.3f} | {ep.get('pixel_auroc', 0):.3f} | "
                     f"{ei.get('recall', 0):.1%} | {ei.get('false_negative_rate', 0):.1%} | "
                     f"{ei.get('false_positive_rate', 0):.1%} |")
    Path(args.out_dir, "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
