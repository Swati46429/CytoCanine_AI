"""
CytoCanine AI
YOLOv8 Result Analysis Script

This script analyzes YOLOv8 prediction outputs and generates:

1. Confidence score distribution
2. Class-wise detection comparison

Usage:
python result_analysis.py
"""

import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Folder containing YOLO prediction label files
LABEL_DIR = "runs/detect/predict/labels"

# Class names (same order as dataset)
CLASS_NAMES = ['Histiocytoma', 'Lymphoma', 'Mast_Cell', 'TVT']


def confidence_distribution():

    conf_scores = []

    for file in os.listdir(LABEL_DIR):

        if not file.endswith(".txt"):
            continue

        with open(os.path.join(LABEL_DIR, file)) as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) == 6:
                    conf_scores.append(float(parts[5]))

    plt.figure(figsize=(8,5))
    plt.hist(conf_scores, bins=10)

    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Detections")

    plt.show()


def class_detection_comparison():

    counts = defaultdict(int)

    for file in os.listdir(LABEL_DIR):

        if not file.endswith(".txt"):
            continue

        with open(os.path.join(LABEL_DIR, file)) as f:

            for line in f:

                cls = int(line.split()[0])
                counts[CLASS_NAMES[cls]] += 1

    plt.figure(figsize=(8,5))
    plt.bar(counts.keys(), counts.values())

    plt.title("Class-wise Detection Counts")
    plt.ylabel("Number of Detections")

    plt.show()


if __name__ == "__main__":

    print("Running result analysis...")

    confidence_distribution()
    class_detection_comparison()

    print("Analysis complete.")
