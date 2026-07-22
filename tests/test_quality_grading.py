"""Unit tests that do not require downloading a YOLO model."""

from app.detector import QualityInspector


def test_grade_no_detections():
    result = QualityInspector._grade_quality([])
    assert result["grade"] == "Needs Review"
    assert result["score"] == 0.0


def test_grade_good():
    detections = [
        {"class_name": "apple", "confidence": 0.91},
        {"class_name": "banana", "confidence": 0.88},
    ]
    result = QualityInspector._grade_quality(detections)
    assert result["grade"] == "Good"
    assert result["score"] > 70


def test_grade_fair():
    detections = [{"class_name": "orange", "confidence": 0.50}]
    result = QualityInspector._grade_quality(detections)
    assert result["grade"] == "Fair"


def test_grade_poor():
    detections = [{"class_name": "apple", "confidence": 0.30}]
    result = QualityInspector._grade_quality(detections)
    assert result["grade"] == "Poor"
