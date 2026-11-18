"""PatchCore — détection et segmentation d'anomalies sans exemple de défaut.

Référence : Roth et al., « Towards Total Recall in Industrial Anomaly Detection », CVPR 2022.

Principe :
1. Un CNN pré-entraîné (ImageNet) extrait des descripteurs locaux (patchs) à mi-profondeur.
2. On mémorise les patchs des pièces SAINES (memory bank), réduits par sous-échantillonnage
   « coreset » glouton pour garder la diversité avec peu de mémoire.
3. En inférence, chaque patch de l'image est comparé à son plus proche voisin dans la
   mémoire : la distance donne une carte d'anomalie (segmentation) et son maximum un score
   global (classification sain / défectueux).

Pourquoi ce choix pour l'aéro : on n'a presque jamais assez d'images de défauts réels
(rares, confidentiels). Un modèle qui apprend « la normalité » détecte aussi des défauts
jamais vus — ce qui est exactement ce qu'on attend d'un premier filtre d'inspection.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PatchCoreConfig:
    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    image_size: int = 256
    feature_dim: int = 1024        # dimension après réduction (moyenne adaptative)
    coreset_ratio: float = 0.1     # fraction de patchs conservés
    projection_dim: int = 128      # projection aléatoire pour accélérer le coreset
    blur_sigma: float = 4.0
    seed: int = 0
    pretrained: bool = True        # False = smoke test hors-ligne (poids aléatoires reproductibles)


@dataclass
class Calibration:
    """Statistiques nécessaires pour passer d'une distance brute à une décision."""
    image_threshold: float = 0.0       # seuil score image (F1 max sur calibration)
    pixel_threshold: float = 0.0       # seuil carte d'anomalie (segmentation)
    platt_a: float = 1.0               # confiance = sigmoid(a * score + b)
    platt_b: float = 0.0
    normal_mean: float = 0.0           # stats des scores sur pièces saines tenues à l'écart
    normal_std: float = 1.0
    source: str = "none"               # "calibration_split" | "normal_only" | "none"

    def confidence(self, score: float) -> float:
        """Probabilité calibrée que la pièce soit défectueuse."""
        z = self.platt_a * score + self.platt_b
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: tuple[str, ...], pretrained: bool = True):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.blocks = nn.ModuleDict({name: getattr(net, name) for name in
                                     ("layer1", "layer2", "layer3", "layer4")})
        self.layers = layers
        self.last = max(layers, key=lambda n: int(n[-1]))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        x = self.stem(x)
        for name, block in self.blocks.items():
            x = block(x)
            if name in self.layers:
                out.append(x)
            if name == self.last:
                break
        return out


def preprocess(images_bgr: list[np.ndarray], size: int) -> torch.Tensor:
    batch = []
    for img in images_bgr:
        rgb = cv2.cvtColor(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        batch.append(torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32)))
    return torch.stack(batch)


def greedy_coreset(features: torch.Tensor, ratio: float, projection_dim: int, seed: int,
                   device: torch.device | str = "cpu", verbose: bool = False) -> torch.Tensor:
    """Sous-échantillonnage k-center glouton (approximation du coreset de PatchCore).

    Calcul sur `device` (GPU si dispo) : ~20 000 itérations sur ~200 000 patchs, c'est
    l'étape la plus longue de l'entraînement.
    """
    n = features.shape[0]
    k = max(1, int(n * ratio))
    if k >= n:
        return torch.arange(n)
    g = torch.Generator().manual_seed(seed)
    if features.shape[1] > projection_dim:
        proj = torch.randn(features.shape[1], projection_dim, generator=g) / projection_dim ** 0.5
        reduced = (features @ proj).to(device)
    else:
        reduced = features.to(device)
    start = int(torch.randint(n, (1,), generator=g))
    selected = [start]
    min_dist = torch.cdist(reduced, reduced[start:start + 1]).squeeze(1)
    step = max(1, k // 10)
    for i in range(k - 1):
        idx = int(torch.argmax(min_dist))
        selected.append(idx)
        min_dist = torch.minimum(min_dist, torch.cdist(reduced, reduced[idx:idx + 1]).squeeze(1))
        if verbose and (i + 2) % step == 0:
            print(f"  coreset {i + 2}/{k} ({(i + 2) / k:.0%})", flush=True)
    return torch.tensor(selected)


class PatchCore:
    def __init__(self, config: PatchCoreConfig | None = None, device: str | None = None,
                 pretrained: bool | None = None):
        self.config = config or PatchCoreConfig()
        if pretrained is not None:
            self.config.pretrained = pretrained
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not self.config.pretrained:
            torch.manual_seed(self.config.seed)   # poids aléatoires mais reproductibles
        self.extractor = FeatureExtractor(self.config.backbone, self.config.layers,
                                          self.config.pretrained).to(self.device)
        self.memory_bank: torch.Tensor | None = None
        self.calibration = Calibration()
        self.category: str = ""
        self.meta: dict = {}

    # ------------------------------------------------------------------ features
    @torch.no_grad()
    def _embed(self, images_bgr: list[np.ndarray]) -> tuple[torch.Tensor, tuple[int, int]]:
        """Retourne (B, H*W, D) descripteurs de patchs et la taille (H, W) de la grille."""
        x = preprocess(images_bgr, self.config.image_size).to(self.device)
        feats = self.extractor(x)
        # agrégation locale (voisinage 3x3) = contexte du patch
        feats = [F.avg_pool2d(f, kernel_size=3, stride=1, padding=1) for f in feats]
        h, w = feats[0].shape[-2:]
        feats = [f if f.shape[-2:] == (h, w) else
                 F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False) for f in feats]
        emb = torch.cat(feats, dim=1)                              # (B, C, H, W)
        b, c = emb.shape[:2]
        emb = emb.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if c > self.config.feature_dim:
            emb = F.adaptive_avg_pool1d(emb, self.config.feature_dim)
        return emb, (h, w)

    # ------------------------------------------------------------------ training
    def fit(self, images_bgr: list[np.ndarray], batch_size: int = 8, verbose: bool = True) -> None:
        t0 = time.time()
        chunks = []
        for i in range(0, len(images_bgr), batch_size):
            emb, _ = self._embed(images_bgr[i:i + batch_size])
            chunks.append(emb.reshape(-1, emb.shape[-1]).cpu())
        all_feats = torch.cat(chunks)
        if verbose:
            print(f"[PatchCore] {all_feats.shape[0]} patchs extraits, sélection du coreset "
                  f"sur {self.device}...", flush=True)
        idx = greedy_coreset(all_feats, self.config.coreset_ratio, self.config.projection_dim,
                             self.config.seed, device=self.device, verbose=verbose)
        self.memory_bank = all_feats[idx].contiguous()
        self.meta.update(n_train_images=len(images_bgr), n_patches=int(all_feats.shape[0]),
                         memory_bank_size=int(self.memory_bank.shape[0]),
                         fit_seconds=round(time.time() - t0, 1))
        if verbose:
            print(f"[PatchCore] {len(images_bgr)} images -> {all_feats.shape[0]} patchs "
                  f"-> mémoire {self.memory_bank.shape[0]} ({time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _nn_distance(self, queries: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        bank = self.memory_bank.to(self.device)
        out = []
        for i in range(0, queries.shape[0], chunk):
            d = torch.cdist(queries[i:i + chunk].to(self.device), bank)
            out.append(d.min(dim=1).values)
        return torch.cat(out)

    def predict(self, images_bgr: list[np.ndarray] | np.ndarray, batch_size: int = 8
                ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Retourne (scores image (B,), cartes d'anomalie à la taille de chaque image d'origine)."""
        if self.memory_bank is None:
            raise RuntimeError("Modèle non entraîné : appeler fit() ou load().")
        if isinstance(images_bgr, np.ndarray):
            images_bgr = [images_bgr]
        scores, maps = [], []
        for i in range(0, len(images_bgr), batch_size):
            batch = images_bgr[i:i + batch_size]
            emb, (h, w) = self._embed(batch)
            b = emb.shape[0]
            dist = self._nn_distance(emb.reshape(-1, emb.shape[-1])).reshape(b, 1, h, w)
            size = self.config.image_size
            up = F.interpolate(dist, size=(size, size), mode="bilinear", align_corners=False)
            for j, img in enumerate(batch):
                amap = up[j, 0].cpu().numpy()
                amap = cv2.GaussianBlur(amap, (0, 0), self.config.blur_sigma)
                amap = cv2.resize(amap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                maps.append(amap.astype(np.float32))
                scores.append(float(dist[j].max()))
        return np.asarray(scores, dtype=np.float32), maps

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "aerovision-patchcore/1",
            "category": self.category,
            "config": asdict(self.config),
            "calibration": asdict(self.calibration),
            "memory_bank": self.memory_bank.half(),   # fp16 : /2 sur disque, perte négligeable
            "meta": self.meta,
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "PatchCore":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = dict(ckpt["config"])
        cfg["layers"] = tuple(cfg["layers"])
        model = cls(PatchCoreConfig(**cfg), device=device)
        model.memory_bank = ckpt["memory_bank"].float()
        model.calibration = Calibration(**ckpt["calibration"])
        model.category = ckpt.get("category", "")
        model.meta = ckpt.get("meta", {})
        return model
