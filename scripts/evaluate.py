"""
Evaluate a YOLO model and write a simple performance report.

Usage:
  python scripts/evaluate.py --model models/yolov8n.pt --data data/dataset.yaml

If you do not have a custom dataset yet, this script writes a demo report
explaining how evaluation works for the portfolio.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "evaluation_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate quality inspection model")
    parser.add_argument("--model", default=str(ROOT / "models" / "yolov8n.pt"))
    parser.add_argument("--data", default=str(ROOT / "data" / "dataset.yaml"))
    return parser.parse_args()


def write_demo_report(model: str, data: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = f"""# Performance Evaluation Report

**Generated:** {now}

## Model
- Weights: `{model}`
- Architecture: YOLOv8n (nano)
- Task: Object detection for visual quality inspection (fruit/food demo)

## Dataset
- Config: `{data}`
- Status: Demo mode (custom labeled dataset optional)

## Metrics Explained (what recruiters look for)
| Metric | Meaning |
|--------|---------|
| mAP50 | Mean Average Precision at IoU 0.50 |
| mAP50-95 | Mean Average Precision across IoU 0.50 to 0.95 |
| Precision | Of predicted boxes, how many are correct |
| Recall | Of true objects, how many we found |

## Demo / Baseline Notes
This project ships with a pretrained COCO YOLOv8n model so the app works immediately.

For a custom manufacturing / fruit-grading dataset:
1. Annotate images in Roboflow (bounding boxes)
2. Export YOLO format into `data/`
3. Update `data/dataset.yaml`
4. Run `python scripts/train.py`
5. Re-run this script with your best weights

## Example command after custom training
```bash
python scripts/evaluate.py --model runs/detect/quality_inspection/weights/best.pt --data data/dataset.yaml
```

## Qualitative Checks
- Upload clear apple/banana/orange photos in Streamlit
- Confirm boxes are drawn and confidence scores appear
- Confirm API `/predict` returns JSON + annotated image URL
"""
    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote report: {REPORT_PATH}")


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        write_demo_report(args.model, args.data)
        return

    try:
        from ultralytics import YOLO

        model = YOLO(args.model)
        metrics = model.val(data=str(data_path))
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        box = getattr(metrics, "box", None)
        map50 = getattr(box, "map50", None) if box else None
        map5095 = getattr(box, "map", None) if box else None
        precision = getattr(box, "mp", None) if box else None
        recall = getattr(box, "mr", None) if box else None

        content = f"""# Performance Evaluation Report

**Generated:** {now}

## Model
- Weights: `{args.model}`

## Dataset
- Config: `{args.data}`

## Results
| Metric | Value |
|--------|-------|
| mAP50 | {map50} |
| mAP50-95 | {map5095} |
| Precision | {precision} |
| Recall | {recall} |
"""
        REPORT_PATH.write_text(content, encoding="utf-8")
        print(f"Wrote report: {REPORT_PATH}")
    except Exception as exc:  # noqa: BLE001
        print(f"Validation failed ({exc}). Writing demo report instead.")
        write_demo_report(args.model, args.data)


if __name__ == "__main__":
    main()
