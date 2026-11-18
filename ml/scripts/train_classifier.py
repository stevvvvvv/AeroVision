"""Entraîne le classifieur de type de pièce (ResNet-18, transfer learning).

    python scripts/train_classifier.py --data-root data/mvtec_ad --epochs 8

Entraînement sur train/good uniquement ; évaluation sur le test (pièces saines ET
défectueuses) : le classifieur doit reconnaître la pièce même abîmée.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aerovision_ml import data  # noqa: E402
from aerovision_ml.classifier import PartClassifier, build_model, to_tensor  # noqa: E402


class PartsDataset(Dataset):
    def __init__(self, samples, classes, augment: bool):
        self.samples, self.idx, self.augment = samples, {c: i for i, c in enumerate(classes)}, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = data.load_image(s.path)
        if self.augment:
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
            angle = np.random.uniform(-20, 20)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, np.random.uniform(0.9, 1.1))
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            img = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.8, 1.2), beta=np.random.uniform(-20, 20))
        return to_tensor(img), self.idx[s.category]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default="models/classifier.pt")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    classes = data.list_categories(args.data_root)
    train = [s for c in classes for s in data.train_samples(args.data_root, c)]
    test = [s for c in classes for s in data.test_samples(args.data_root, c)]

    model = build_model(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    tl = DataLoader(PartsDataset(train, classes, True), batch_size=args.batch_size, shuffle=True, num_workers=2)
    vl = DataLoader(PartsDataset(test, classes, False), batch_size=args.batch_size, num_workers=2)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(train):.4f}")

    model.eval()
    correct, confusion = 0, np.zeros((len(classes), len(classes)), int)
    with torch.no_grad():
        for x, y in vl:
            pred = model(x.to(device)).argmax(1).cpu()
            correct += int((pred == y).sum())
            for t, pr in zip(y.tolist(), pred.tolist()):
                confusion[t, pr] += 1
    acc = correct / len(test)
    print(f"Précision test (pièces saines + défectueuses) : {acc:.2%}")

    clf = PartClassifier(model.cpu(), classes, "cpu",
                         meta={"test_accuracy": acc, "n_train": len(train), "n_test": len(test)})
    clf.save(args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"classes": classes, "test_accuracy": acc, "confusion": confusion.tolist()}, indent=2))


if __name__ == "__main__":
    main()
