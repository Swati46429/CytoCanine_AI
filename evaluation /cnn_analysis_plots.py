"""
CytoCanine AI
CNN Result Analysis Script

This script generates important analysis plots for the trained
ConvNeXt-Tiny tumor classification model.

Generated Plots:
1. Training vs Validation Loss Curve
2. Training vs Validation Accuracy Curve
3. Multi-class ROC Curve with AUC

Usage:
python analysis_plots.py
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# ===============================
# PATH SETTINGS
# ===============================

MODEL_PATH = "models/convnext_tiny_final_earlystop.pth"
TEST_DATASET = "dataset/test"

# ===============================
# TRAINING HISTORY (FROM TRAINING)
# ===============================

train_loss = [0.7878,0.3764,0.2928,0.2476,0.2215,0.2024,0.1886,0.1788,0.1705,0.1616,
              0.1567,0.1526,0.1454,0.1410,0.1389,0.1348,0.1306,0.1297,0.1270,0.1240,
              0.1219,0.1204,0.1183,0.1179,0.1165,0.1140,0.1104,0.1121,0.1086,0.1070]

valid_loss = [0.3937,0.2680,0.2192,0.1917,0.1776,0.1584,0.1512,0.1393,0.1347,0.1294,
              0.1242,0.1219,0.1189,0.1155,0.1123,0.1111,0.1102,0.1104,0.1040,0.1035,
              0.1047,0.1025,0.1001,0.0968,0.0996,0.0997,0.0984,0.0979,0.0943,0.0930]

train_acc = [0.7934,0.8805,0.8989,0.9101,0.9204,0.9253,0.9298,0.9322,0.9365,0.9397,
             0.9409,0.9426,0.9442,0.9468,0.9477,0.9487,0.9497,0.9505,0.9506,0.9534,
             0.9546,0.9540,0.9543,0.9558,0.9552,0.9563,0.9580,0.9560,0.9572,0.9585]

valid_acc = [0.8786,0.9038,0.9179,0.9262,0.9333,0.9406,0.9427,0.9475,0.9491,0.9506,
             0.9530,0.9544,0.9560,0.9587,0.9587,0.9595,0.9601,0.9591,0.9613,0.9619,
             0.9623,0.9619,0.9625,0.9637,0.9627,0.9625,0.9621,0.9621,0.9631,0.9635]

# ===============================
# PLOT TRAINING CURVES
# ===============================

def plot_training_curves():

    epochs = np.arange(1, len(train_loss) + 1)

    # LOSS CURVE
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label="Train Loss", marker='o')
    plt.plot(epochs, valid_loss, label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ACCURACY CURVE
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
    plt.plot(epochs, valid_acc, label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===============================
# ROC CURVE
# ===============================

def plot_roc_curve():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    dataset = datasets.ImageFolder(TEST_DATASET, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    class_names = dataset.classes
    num_classes = len(class_names)

    # Load model
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    y_true = []
    y_scores = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            y_scores.extend(probs.cpu().numpy())
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    plt.figure(figsize=(8,6))

    for i in range(num_classes):

        fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_scores[:,i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve (ConvNeXt-Tiny)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":

    print("Generating training curves...")
    plot_training_curves()

    print("Generating ROC curve...")
    plot_roc_curve()
