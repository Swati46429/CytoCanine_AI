# CytoCanine AI

AI-assisted detection and classification of canine round cell tumors (RCTs) from cytological images using deep learning.

This repository contains the source code for the **CytoCanine AI system**, developed as part of the study:

**"AI-Assisted Cytological Detection and Classification of Canine Round Cell Tumors"**

---

# Project Overview

CytoCanine AI is an artificial intelligence–based diagnostic framework designed to assist in the detection and classification of **canine round cell tumors (RCTs)** from cytological microscopic images.

The proposed pipeline integrates two deep learning models:

1. **YOLOv8x** – Tumor region detection
2. **ConvNeXt-Tiny** – Multi-class tumor classification
3. **Grad-CAM** – Explainable AI visualization for model interpretability

The system analyzes **high-power microscopic cytology images** and identifies tumor regions followed by classification into specific tumor subtypes.

---

# Tumor Classes

The classification model predicts the following tumor categories:

* Histiocytoma
* Lymphoma
* Mast Cell Tumor
* Transmissible Venereal Tumor (TVT)
* Negative (Non-round cell tumor)

---

# Live Demo

Interactive web application available at:

https://huggingface.co/spaces/DeepBioSwati/CytoCanine_AI

Users can upload cytology images and obtain automated tumor detection and classification results.

---

# Model Weights

Due to GitHub file size limitations, trained model weights are hosted on Hugging Face.

Model repository:

https://huggingface.co/DeepBioSwati/CytoCanine_AI_models

Available files:

* `yolov8x_best.pt` – trained YOLOv8 detection model
* `convnext_tiny_final_earlystop.pth` – trained ConvNeXt-Tiny classification model

Download the models and place them inside the **models/** directory before running inference.

---

# Repository Structure

```
CytoCanine_AI
│
├── training_code
│   ├── dataset_preparation.py
│   ├── augmentation.py
│   └── train_yolov8.py
│
├── inference
│   └── predict_image.py
│
├── evaluation
│   ├── evaluate_yolo.py
│   └── result_analysis.py
│
├── models
│
├── app.py
├── README.md
├── requirements.txt
├── runtime.txt
└── packages.txt
```

---

# Pipeline Workflow

The CytoCanine AI workflow follows this pipeline:

Dataset Preparation
↓
Data Augmentation
↓
YOLOv8 Model Training
↓
Model Evaluation
↓
Inference on New Cytology Images
↓
Result Analysis and Visualization

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

# Model Training

To train the YOLOv8 tumor detection model:

```
python training_code/train_yolov8.py
```

Dataset structure expected:

```
dataset/
 ├── train
 │   ├── images
 │   └── labels
 ├── valid
 │   ├── images
 │   └── labels
 └── data.yaml
```

---

# Model Evaluation

Evaluate the trained YOLOv8 model:

```
python evaluation/evaluate_yolo.py
```

This script computes detection metrics including:

* Precision
* Recall
* mAP@0.5
* mAP@0.5:0.95
* Confusion Matrix

---

# Inference (Prediction)

Run tumor detection on a new cytology image:

```
python inference/predict_image.py
```

The script loads the trained YOLOv8 model and predicts tumor regions from the input image.

---

# Result Analysis

Research plots and statistical analysis can be generated using:

```
python evaluation/result_analysis.py
```

This includes visualization of:

* Class-wise detection performance
* Confidence score distributions
* Detection comparison plots

These plots correspond to the **results presented in the research study**.

---

# Application Interface

The repository also contains the **Gradio-based web application** used for interactive inference.

Run locally:

```
python app.py
```

---

# Citation

If you use this repository in your research, please cite:

CytoCanine AI – AI-Assisted Cytological Detection and Classification of Canine Round Cell Tumors.

---

# License

This project is released under the **MIT License**.
