"""
CytoCanine AI
Inference Script for YOLOv8 Tumor Detection

This script loads the trained YOLOv8x model and performs tumor detection
on input cytology images.

Usage:
python predict_image.py --image path/to/image.jpg
"""

import argparse
from ultralytics import YOLO


def run_inference(image_path):

    print("Loading trained YOLOv8 model...")

    model = YOLO("models/yolov8Vx_best.pt")

    print("Running prediction...")

    results = model.predict(
        source=image_path,
        conf=0.5,
        save=True,
        save_txt=True
    )

    print("Prediction completed.")
    print("Results saved in runs/detect folder.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input cytology image"
    )

    args = parser.parse_args()

    run_inference(args.image)
