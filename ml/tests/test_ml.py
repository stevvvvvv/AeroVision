"""Tests du pipeline ML (sans téléchargement : backbone non pré-entraîné + données synthétiques)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aerovision_ml import baseline, data, taxonomy  # noqa: E402
from aerovision_ml.calibration import best_f1_threshold, fit_platt  # noqa: E402
from aerovision_ml.patchcore import PatchCore, PatchCoreConfig, greedy_coreset  # noqa: E402
from aerovision_ml.postprocess import extract_regions, iou, normalize_map, render_overlay  # noqa: E402


@pytest.fixture(scope="session")
def synth(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic_dataset.py"),
                    "--out", str(out), "--n-train", "12", "--n-test", "6"], check=True)
    return out


def test_taxonomy_is_complete():
    for cat in taxonomy.MVTEC_CATEGORIES:
        assert cat in taxonomy.MVTEC_TO_GENERIC
        assert cat in taxonomy.AERO_PROXY_RELEVANCE
        for g in taxonomy.MVTEC_TO_GENERIC[cat].values():
            assert g in taxonomy.GENERIC_DEFECTS
    for g in taxonomy.GENERIC_DEFECTS:
        assert g in taxonomy.GENERIC_TO_AERO_HINT


def test_extract_regions_finds_blob():
    amap = np.zeros((100, 100), np.float32)
    amap[40:60, 20:35] = 1.0
    regions, mask = extract_regions(amap, 0.5)
    assert len(regions) == 1
    r = regions[0]
    assert abs(r.x - 20) <= 2 and abs(r.y - 40) <= 2
    assert r.max_score == pytest.approx(1.0)
    assert mask.max() == 255


def test_extract_regions_ignores_noise():
    amap = np.zeros((100, 100), np.float32)
    amap[10, 10] = 1.0
    regions, _ = extract_regions(amap, 0.5)
    assert regions == []


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_calibration_helpers():
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    thr, f1 = best_f1_threshold(scores, labels)
    assert f1 == pytest.approx(1.0) and 0.3 < thr <= 0.8
    a, b = fit_platt(scores, labels)
    assert a > 0


def test_coreset_size():
    x = np.random.default_rng(0).normal(size=(500, 32)).astype(np.float32)
    import torch
    idx = greedy_coreset(torch.from_numpy(x), 0.1, 16, 0)
    assert len(idx) == 50 and len(set(idx.tolist())) == 50


def test_baseline_detects_synthetic_crack(synth):
    tests = data.test_samples(synth, "synthetic_plate")
    good = [s for s in tests if not s.is_anomalous]
    crack = [s for s in tests if s.defect == "hole"]
    g = baseline.image_score(baseline.anomaly_map(data.load_image(good[0].path)))
    c = baseline.image_score(baseline.anomaly_map(data.load_image(crack[0].path)))
    assert c > g


def test_patchcore_end_to_end(synth, tmp_path):
    train = [data.load_image(s.path) for s in data.train_samples(synth, "synthetic_plate")]
    cfg = PatchCoreConfig(backbone="resnet18", feature_dim=128, image_size=128, coreset_ratio=0.2)
    model = PatchCore(cfg, device="cpu", pretrained=False)
    model.fit(train, verbose=False)
    tests = data.test_samples(synth, "synthetic_plate")
    scores, maps = model.predict([data.load_image(s.path) for s in tests])
    assert len(scores) == len(tests)
    assert maps[0].shape == data.load_image(tests[0].path).shape[:2]

    model.save(tmp_path / "m.pt")
    reloaded = PatchCore.load(tmp_path / "m.pt", device="cpu")
    s2, _ = reloaded.predict([data.load_image(tests[0].path)])
    assert s2[0] == pytest.approx(scores[0], rel=1e-2)  # fp16 en stockage

    over = render_overlay(data.load_image(tests[0].path), normalize_map(maps[0], 0, scores.max()), [])
    assert over.shape == data.load_image(tests[0].path).shape
