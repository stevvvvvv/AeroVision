"""Tests d'API de bout en bout (mode repli OpenCV, aucun modèle requis)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("api")
    os.environ["AEROVISION_DATABASE_URL"] = f"sqlite:///{tmp}/t.db"
    os.environ["AEROVISION_STORAGE_DIR"] = str(tmp / "storage")
    os.environ["AEROVISION_MODELS_DIR"] = str(tmp / "models")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def _png(defect: bool) -> bytes:
    rng = np.random.default_rng(1)
    img = np.clip(160 + rng.normal(0, 3, (256, 256)), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if defect:
        cv2.line(img, (60, 80), (180, 150), (30, 30, 30), 3)
    return cv2.imencode(".png", img)[1].tobytes()


def test_health_and_system(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    info = client.get("/api/system").json()
    assert info["mode"] == "baseline_opencv"
    assert "crack" in info["defect_classes"]


def test_full_flow(client):
    r = client.post("/api/inspections", files={"file": ("p.png", _png(True), "image/png")},
                    data={"part_reference": "PN-123"})
    assert r.status_code == 201, r.text
    insp = r.json()
    assert insp["engine"] == "baseline_opencv"
    assert insp["status"] == "pending_review"
    assert insp["n_detections"] >= 1
    assert any("OpenCV" in w for w in insp["warnings"])
    d = insp["detections"][0]
    # la boîte recouvre la fissure tracée
    assert d["x"] < 120 < d["x"] + d["w"]

    for kind in ("original", "heatmap", "overlay", "mask"):
        assert client.get(f"/api/inspections/{insp['id']}/image/{kind}").status_code == 200

    # incohérence refusée
    bad = client.post(f"/api/inspections/{insp['id']}/reviews", json={
        "reviewer": "S. Jansen", "decision": "conforme",
        "detection_feedback": [{"detection_id": d["id"], "is_true_defect": True}]})
    assert bad.status_code == 422

    ok = client.post(f"/api/inspections/{insp['id']}/reviews", json={
        "reviewer": "S. Jansen", "decision": "non_conforme", "comment": "Fissure confirmée",
        "detection_feedback": [{"detection_id": d["id"], "is_true_defect": True, "corrected_type": "crack"}],
        "added_regions": [{"x": 10, "y": 10, "w": 20, "h": 20, "defect_type": "scratch"}]})
    assert ok.status_code == 201
    assert ok.json()["status"] == "reviewed"
    assert len(ok.json()["reviews"]) == 1

    stats = client.get("/api/stats").json()
    assert stats["detection_feedback"]["confirmed"] == 1
    assert stats["detection_feedback"]["missed_regions_added"] == 1

    csv_text = client.get("/api/export/feedback.csv").text
    assert "Fissure confirmée" in csv_text and "missed" in csv_text


def test_queue_and_filters(client):
    client.post("/api/inspections", files={"file": ("ok.png", _png(False), "image/png")})
    page = client.get("/api/inspections", params={"status": "pending_review", "queue": True}).json()
    assert page["total"] >= 1


def test_rejects_non_image(client):
    r = client.post("/api/inspections", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415
