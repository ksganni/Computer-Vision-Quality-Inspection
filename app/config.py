"""Simple settings for the quality inspection app."""

from pathlib import Path

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

# Default YOLO model (small + fast). Auto-downloads on first run.
DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_MODEL_PATH = MODELS_DIR / DEFAULT_MODEL_NAME

# Confidence threshold for showing detections
CONFIDENCE_THRESHOLD = 0.25

# Fruit / food related COCO classes we care about for this demo
TARGET_CLASSES = {
    "apple",
    "banana",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "sandwich",
}

# Make sure folders exist when the app starts
for folder in (MODELS_DIR, UPLOADS_DIR, OUTPUTS_DIR, SAMPLE_IMAGES_DIR):
    folder.mkdir(parents=True, exist_ok=True)
