# CytoCanine AI

AI-assisted detection and classification of canine round cell tumors (RCTs) from cytological images using deep learning.

This repository contains the source code for the CytoCanine AI web application developed as part of the study:

"AI-Assisted Cytological Detection and Classification of Canine Round Cell Tumors"

---

## Project Overview

The proposed pipeline integrates:

1. YOLOv8x – Tumor region detection  
2. ConvNeXt-Tiny – Multi-class tumor classification  
3. Grad-CAM – Explainable AI visualization

The system analyzes high-power cytological microscopic fields to identify and classify round cell tumor subtypes.

---

## Tumor Classes

The model classifies cytological images into the following categories:

- Histiocytoma
- Lymphoma
- Mast Cell Tumor
- Transmissible Venereal Tumor (TVT)
- Negative (non-round cell tumor)

---

## Live Demo

Interactive web application:

https://huggingface.co/spaces/DeepBioSwati/CytoCanine_AI

---

## Model Weights

Due to GitHub file size limitations, trained model weights are hosted on Hugging Face:

https://huggingface.co/DeepBioSwati/CytoCanine_AI_models

Files available:

- yolov8x_best.pt
- convnext_tiny_final_earlystop.pth

---

## Repository Structure
