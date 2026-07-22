# Performance Evaluation Report

**Generated:** 2026-07-18 03:49 UTC

## Model
- Weights: `/Users/krishnasathvikaganni/Projects/computer-vision-quality-inspection/models/yolov8n.pt`
- Architecture: YOLOv8n (nano)
- Task: Object detection for visual quality inspection (fruit/food demo)

## Dataset
- Config: `/Users/krishnasathvikaganni/Projects/computer-vision-quality-inspection/data/dataset.yaml`
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
