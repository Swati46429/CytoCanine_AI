"""
CytoCanine AI
YOLOv8 Model Evaluation Script

This script evaluates the trained YOLOv8 model on a validation dataset
and computes detection metrics including Precision, Recall, and mAP.

Usage:
python evaluate_yolo.py
"""

from ultralytics import YOLO


def evaluate_model():

    print("Loading trained YOLOv8 model...")

    model = YOLO("models/yolov8Vx_best.pt")

    print("Running validation...")

    metrics = model.val(
        data="dataset/data.yaml",
        imgsz=640,
        conf=0.25,
        iou=0.5,
        plots=True
    )

    print("Evaluation complete.")
    print(metrics)


if __name__ == "__main__":

    evaluate_model()
