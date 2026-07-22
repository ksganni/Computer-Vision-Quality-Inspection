"""
Simple YOLOv8 training entrypoint.

Usage (after you add a real labeled dataset):
  python scripts/train.py --data data/dataset.yaml --epochs 20 --imgsz 640

For this portfolio project, training is optional.
The demo works out of the box with the pretrained yolov8n model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 quality inspection model")
    parser.add_argument("--data", default=str(ROOT / "data" / "dataset.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    parser.add_argument("--name", default="quality_inspection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"Dataset config not found: {data_path}\n"
            "Add your Roboflow/YOLO labels and point dataset.yaml to them."
        )

    print(f"Starting training with data={data_path}")
    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print("Training finished. Check the runs/ folder for weights and charts.")


if __name__ == "__main__":
    main()
