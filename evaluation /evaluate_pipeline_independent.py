"""
Evaluate full CytoCanine AI pipeline
YOLOv8 detection + ConvNeXt classification

Runs on independent dataset (276 images)
and generates confusion matrix + accuracy.
"""

import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F

from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


# ================= SETTINGS =================

YOLO_MODEL_PATH = "models/yolov8Vx_best.pt"
CNN_MODEL_PATH = "models/convnext_tiny_final_earlystop.pth"

TEST_DIR = "independent_test"

CLASS_NAMES = [
    "Histiocytoma",
    "Lymphoma",
    "Mast_cell_Tumor",
    "Negative",
    "TVT"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_PATCHES = 5
LYMPHOMA_THRESHOLD = 0.7
CONF_THRESHOLD = 0.6
YOLO_OVERRIDE_THRESHOLD = 3


# ================= TRANSFORM =================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# ================= LOAD MODELS =================

print("Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_class_names = list(yolo_model.names.values())

print("Loading ConvNeXt...")
cnn_model = models.convnext_tiny(weights=None)

num_features = cnn_model.classifier[2].in_features
cnn_model.classifier[2] = torch.nn.Linear(num_features, len(CLASS_NAMES))

cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.to(device)
cnn_model.eval()

print("Models loaded successfully\n")


# ================= PIPELINE =================

def pipeline_predict(image_path):

    img_bgr = cv2.imread(image_path)

    # Safety check
    if img_bgr is None:
        print(f"Warning: could not read image {image_path}")
        return "Negative"

    results = yolo_model(img_bgr, conf=0.45, imgsz=800, verbose=False)[0]

    patch_predictions=[]
    yolo_detected_classes=[]
    patch_count=0

    for box in results.boxes:

        x1,y1,x2,y2 = map(int,box.xyxy[0])
        crop = img_bgr[y1:y2,x1:x2]

        if crop.size == 0:
            continue

        cls_id=int(box.cls[0])
        if 0<=cls_id<len(yolo_class_names):
            yolo_detected_classes.append(
                yolo_class_names[cls_id])

        crop_pil = Image.fromarray(
            cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))

        input_tensor = transform(crop_pil)\
                        .unsqueeze(0).to(device)

        with torch.no_grad():
            output = cnn_model(input_tensor)
            probs = F.softmax(output,dim=1).squeeze()

        pred_idx = torch.argmax(probs).item()
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = probs[pred_idx].item()

        if pred_class=="Histiocytoma" and pred_conf>=0.5:
            pred_conf=min(pred_conf+0.1,1.0)

        if pred_class=="Mast_cell_Tumor" and pred_conf>=0.45:
            patch_predictions.append((pred_class,pred_conf))
            patch_count+=1

        elif pred_conf>=CONF_THRESHOLD:
            patch_predictions.append((pred_class,pred_conf))
            patch_count+=1

    yolo_counter = Counter(yolo_detected_classes)

    cnn_prediction="Negative"
    cnn_confidence=0.0

    if patch_count>=MIN_PATCHES:

        vote_counts=np.zeros(len(CLASS_NAMES),dtype=int)
        conf_sums=np.zeros(len(CLASS_NAMES),dtype=float)

        for cls,conf in patch_predictions:

            idx=CLASS_NAMES.index(cls)
            vote_counts[idx]+=1
            conf_sums[idx]+=conf

        avg_conf=conf_sums/(vote_counts+1e-6)

        lymphoma_ratio = vote_counts[
            CLASS_NAMES.index("Lymphoma")
        ] / patch_count

        if lymphoma_ratio>=LYMPHOMA_THRESHOLD:

            cnn_prediction="Lymphoma"
            cnn_confidence=lymphoma_ratio

        else:

            combined_score = vote_counts + avg_conf*2.5
            top_idx=np.argmax(combined_score)

            cnn_prediction=CLASS_NAMES[top_idx]
            cnn_confidence=avg_conf[top_idx]

    # YOLO override

    if yolo_counter.get("Histiocytoma",0)>=15:
        cnn_prediction="Histiocytoma"

    elif yolo_counter.get("Lymphoma",0)>=YOLO_OVERRIDE_THRESHOLD:
        cnn_prediction="Lymphoma"

    elif yolo_counter.get("Mast cell",0)>=5:
        cnn_prediction="Mast_cell_Tumor"

    elif yolo_counter.get("TVT",0)>=YOLO_OVERRIDE_THRESHOLD:
        cnn_prediction="TVT"

    return cnn_prediction


# ================= EVALUATION =================

y_true=[]
y_pred=[]

print("Running evaluation...\n")

for class_idx, class_name in enumerate(CLASS_NAMES):

    class_folder = os.path.join(TEST_DIR, class_name)

    for img_name in os.listdir(class_folder):

        # image filter
        if not img_name.lower().endswith((".jpg",".jpeg",".png")):
            continue

        img_path = os.path.join(class_folder, img_name)

        pred = pipeline_predict(img_path)

        y_true.append(class_idx)
        y_pred.append(CLASS_NAMES.index(pred))


# ================= METRICS =================

accuracy = accuracy_score(y_true,y_pred)

print("\nOverall Accuracy:", round(accuracy*100,2), "%")

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES
))


# ================= CONFUSION MATRIX =================

cm = confusion_matrix(y_true,y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Independent Dataset")

plt.tight_layout()

os.makedirs("results",exist_ok=True)

plt.savefig("results/confusion_matrix_independent_pipeline.png")

plt.show()

print("\nConfusion matrix saved to results/")
