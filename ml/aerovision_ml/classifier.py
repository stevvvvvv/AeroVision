"""Classification du type de pièce (CNN, transfer learning ResNet-18).

Rôle dans le pipeline : identifier la pièce pour la router vers le modèle d'anomalie
correspondant (un PatchCore par référence de pièce, comme en production où chaque
référence a son propre « golden sample »).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torch import nn

from .patchcore import IMAGENET_MEAN, IMAGENET_STD

INPUT_SIZE = 224


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    net = torchvision.models.resnet18(weights="DEFAULT" if pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32))


class PartClassifier:
    def __init__(self, model: nn.Module, classes: list[str], device: str | None = None,
                 meta: dict | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device).eval()
        self.classes = classes
        self.meta = meta or {}

    @torch.no_grad()
    def predict(self, image_bgr: np.ndarray) -> tuple[str, float, dict[str, float]]:
        x = to_tensor(image_bgr).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        i = int(probs.argmax())
        top = {self.classes[j]: float(probs[j]) for j in probs.argsort()[::-1][:3]}
        return self.classes[i], float(probs[i]), top

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "aerovision-classifier/1", "classes": self.classes,
                    "state_dict": self.model.state_dict(), "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "PartClassifier":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(len(ckpt["classes"]), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["classes"], device, ckpt.get("meta"))
