# CytoCanine AI

AI-assisted detection and classification of canine round cell tumors (RCTs) from cytological images using deep learning.

This repository contains the complete implementation of the **CytoCanine AI system**, developed as part of the research study:

**"Computer Vision Technology-Assisted Microscopic Detection of Round Cell Tumors in Dogs"**

---

# Project Overview

CytoCanine AI is a two-stage deep learning framework designed to assist in the **automated detection and classification of canine round cell tumors (RCTs)** from cytological microscopic images.

The pipeline integrates:

1. **YOLOv8x** – Tumor region detection (object detection)
2. **ConvNeXt-Tiny (CNN)** – Multi-class tumor classification
3. **Grad-CAM** – Explainable AI for visual interpretation

The system processes **high-resolution cytology images**, detects tumor regions, and classifies them into clinically relevant tumor categories.

---

# Tumor Classes

The classification model predicts:

* Histiocytoma
* Lymphoma
* Mast Cell Tumor
* Transmissible Venereal Tumor (TVT)
* Negative (non-round cell tumor)

---

# Dataset

The dataset used in this study is publicly available on Kaggle:

👉 https://www.kaggle.com/datasets/swatijaiswal46429/cytocanine-ai-dataset-yolo-cnn-cropped

It consists of three components:

### 1. YOLO Detection Dataset (Tumor-Detection-13)

* Annotated cytology images
* Bounding box labels for tumor regions
* Used for training YOLOv8

### 2. CNN Cropped Dataset (Cropped-Set)

* Generated from YOLO annotations
* Cropped tumor regions resized to 320×320
* Used for classification model training

### 3. Independent Test Dataset (276 images)

* Completely unseen dataset
* Used for final evaluation and reporting results

---

# Live Demo

Interactive web application:

👉 https://huggingface.co/spaces/DeepBioSwati/CytoCanine_AI

Upload cytology images and get automated predictions.

---

# Model Weights

Pretrained models are available on Hugging Face:

👉 https://huggingface.co/DeepBioSwati/CytoCanine_AI_models

Download and place inside:

```
models/
```

Available weights:

* `yolov8x_best.pt` – YOLO detection model
* `convnext_tiny_final_earlystop.pth` – CNN classification model

---

# Repository Structure

```
CytoCanine_AI
│
├── training_code/
│   ├── cnn/
│   │   ├── train_convnext.py
│   │   ├── dataset_preparation.py
│   │   └── augmentation_pipeline.py
│   │
│   └── yolo/
│       ├── train_yolov8.py
│       ├── dataset_preparation.py
│       └── augmentation.py
│
├── inference/
│   ├── predict_pipeline.py
│   └── predict_yolo.py
│
├── evaluation/
│   ├── evaluate_pipeline_independent.py
│   ├── evaluate_cnn.py
│   ├── evaluate_yolo.py
│   ├── cnn_analysis_plots.py
│   └── yolo_analysis_plots.py
│
├── models/
│   └── placeholder.txt
│
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
└── packages.txt
```

---

# Pipeline Workflow

The complete pipeline:

```
Input Cytology Image
        ↓
YOLOv8 → Detect tumor regions (bounding boxes)
        ↓
Patch Extraction (cropping)
        ↓
ConvNeXt-Tiny → Classify tumor type
        ↓
Aggregation Logic (patch-level → image-level)
        ↓
Final Prediction
        ↓
Grad-CAM Visualization (optional)
```

---

# Installation

Clone the repository:

```
git clone https://github.com/Swati46429/CytoCanine_AI.git
cd CytoCanine_AI
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Training

## YOLO Training

```
python training_code/yolo/train_yolov8.py
```

Expected dataset format:

```
dataset/
 ├── train/images
 ├── train/labels
 ├── valid/images
 ├── valid/labels
 └── data.yaml
```

---

## CNN Training

```
python training_code/cnn/train_convnext.py
```

Expected dataset format:

```
Cropped-Set/
 ├── train/
 ├── valid/
 └── test/
```

---

# Evaluation

## Full Pipeline (Recommended)

```
python evaluation/evaluate_pipeline_independent.py
```

Outputs:

* Accuracy
* Classification report
* Confusion matrix

---

## Individual Evaluation

YOLO:

```
python evaluation/evaluate_yolo.py
```

CNN:

```
python evaluation/evaluate_cnn.py
```

---

# Inference

Run full pipeline:

```
python inference/predict_pipeline.py --image path/to/image.jpg
```

Outputs:

* Final tumor prediction
* Confidence score
* YOLO detections
* Optional Grad-CAM visualization

---

# Application Interface

Run the web app locally:

```
python app.py
```

---

# Key Features

* Two-stage detection + classification pipeline
* Patch-based tumor classification
* Hybrid decision logic (CNN + YOLO override)
* Explainable AI using Grad-CAM
* Independent dataset evaluation
* Research-ready implementation

---

# Citation

If you use this work, please cite:

```bibtex
@misc{CytoCanineAI_Dataset2025,
  title={CytoCanine AI Dataset: YOLO-Based Detection and CNN-Based Classification of Canine Round Cell Tumors},
  author={DeepBioSwati},
  year={2025},
  howpublished={\url{https://www.kaggle.com/datasets/swatijaiswal46429/cytocanine-ai-dataset-yolo-cnn-cropped}},
  note={Kaggle dataset for canine cytology image analysis}
}
```

# License

This project is licensed under the **MIT License**.
