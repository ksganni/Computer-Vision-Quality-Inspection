"""API tests with a fake detector (no heavy model download in CI)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app


class FakeInspector:
    def inspect(self, image_path):
        return {
            "image_name": str(image_path).split("/")[-1],
            "num_detections": 1,
            "detections": [
                {
                    "class_name": "apple",
                    "confidence": 0.93,
                    "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 100},
                }
            ],
            "quality_grade": "Good",
            "quality_score": 93.0,
            "quality_notes": "Clear detections with high confidence.",
            "annotated_image": np.zeros((64, 64, 3), dtype=np.uint8),
        }

    def save_annotated_image(self, annotated_image, output_path):
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-image")
        return path


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"].lower()


def test_predict_with_fake_model(monkeypatch):
    monkeypatch.setattr(main_module, "inspector", FakeInspector())
    client = TestClient(app)

    files = {"file": ("apple.jpg", b"fake-bytes", "image/jpeg")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["num_detections"] == 1
    assert data["quality_grade"] == "Good"
    assert data["detections"][0]["class_name"] == "apple"


def test_predict_rejects_non_image():
    client = TestClient(app)
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
