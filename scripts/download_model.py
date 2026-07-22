"""Download the default YOLOv8 model into ./models."""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
TARGET = MODELS / "yolov8n.pt"


def main() -> None:
    if TARGET.exists():
        print(f"Model already exists: {TARGET}")
        return

    print("Downloading yolov8n.pt (small, fast model)...")
    # This downloads weights (cached by ultralytics / local file)
    YOLO("yolov8n.pt")

    # Ultralytics often places yolov8n.pt in the current working directory
    local = Path("yolov8n.pt")
    if local.exists():
        shutil.move(str(local), str(TARGET))
        print(f"Saved model to: {TARGET}")
    else:
        print("Model downloaded into ultralytics cache.")
        print("The app will still load it automatically on first run.")
    print("Done.")


if __name__ == "__main__":
    main()
