"""YOLOv8 object detection helper for quality inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    TARGET_CLASSES,
)


class QualityInspector:
    """Loads a YOLO model and runs defect/object inspection on images."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        confidence: float = CONFIDENCE_THRESHOLD,
        target_classes: set[str] | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.confidence = confidence
        self.target_classes = target_classes if target_classes is not None else TARGET_CLASSES
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """Load a local model, or let ultralytics download yolov8n.pt."""
        # Lazy import so unit tests can run without installing torch/ultralytics
        from ultralytics import YOLO

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if self.model_path.exists():
            return YOLO(str(self.model_path))
        # Ultralytics downloads the named model automatically on first use
        return YOLO(DEFAULT_MODEL_NAME)


    def inspect(
        self,
        image_path: str | Path,
        *,
        detect_all: bool = False,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """Run detection and return predictions + annotated image path info."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        conf = self.confidence if confidence is None else confidence
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
        )
        result = results[0]

        detections: list[dict[str, Any]] = []
        names = result.names or {}
        active_targets = None if detect_all else self.target_classes

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = names.get(class_id, str(class_id))
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                # Keep only target classes when filtering is on
                if active_targets and class_name not in active_targets:
                    continue

                detections.append(
                    {
                        "class_name": class_name,
                        "confidence": round(score, 4),
                        "bbox": {
                            "x1": round(x1, 2),
                            "y1": round(y1, 2),
                            "x2": round(x2, 2),
                            "y2": round(y2, 2),
                        },
                    }
                )

        annotated = result.plot()  # BGR numpy image with boxes drawn
        quality = self._grade_quality(detections)

        return {
            "image_name": image_path.name,
            "num_detections": len(detections),
            "detections": detections,
            "quality_grade": quality["grade"],
            "quality_score": quality["score"],
            "quality_notes": quality["notes"],
            "annotated_image": annotated,
        }

    def save_annotated_image(self, annotated_image: np.ndarray, output_path: str | Path) -> Path:
        """Save the image with boxes drawn on it."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated_image)
        return output_path

    @staticmethod
    def _grade_quality(detections: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Simple demo grading logic for portfolio purposes.

        - No detections  -> Needs Review
        - Avg confidence high and objects found -> Good
        - Low confidence detections -> Fair / Poor
        """
        if not detections:
            return {
                "grade": "Needs Review",
                "score": 0.0,
                "notes": "No target objects found. Try a clearer fruit/food photo.",
            }

        avg_conf = sum(d["confidence"] for d in detections) / len(detections)
        score = round(avg_conf * 100, 1)

        if avg_conf >= 0.70:
            grade = "Good"
            notes = "Clear detections with high confidence."
        elif avg_conf >= 0.45:
            grade = "Fair"
            notes = "Objects found, but confidence is mixed. Recheck lighting/angle."
        else:
            grade = "Poor"
            notes = "Low-confidence detections. Image may be blurry or unclear."

        return {"grade": grade, "score": score, "notes": notes}
