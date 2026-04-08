"""
CytoCanine AI - Final Inference Pipeline
YOLOv8 + ConvNeXt + Grad-CAM

Usage:
python predict_pipeline.py --image path/to/image.jpg
"""

import os
import cv2
import torch
import argparse
import numpy as np
import torch.nn.functional as F

from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
from collections import Counter


# ================= SETTINGS =================

YOLO_MODEL_PATH = "models/yolov8Vx_best.pt"
CNN_MODEL_PATH = "models/convnext_tiny_final_earlystop.pth"

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
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ================= GRAD-CAM =================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x, class_idx):
        output = self.model(x)
        self.model.zero_grad()

        score = output[0, class_idx]
        score.backward()

        grads = self.gradients
        acts = self.activations

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224, 224),
                            mode="bilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


# ================= LOAD MODELS =================

def load_models():

    print("Loading YOLO...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_class_names = list(yolo_model.names.values())

    print("Loading ConvNeXt...")
    cnn_model = models.convnext_tiny(weights=None)

    num_features = cnn_model.classifier[2].in_features
    cnn_model.classifier[2] = torch.nn.Linear(num_features, len(CLASS_NAMES))

    cnn_model.load_state_dict(
        torch.load(CNN_MODEL_PATH, map_location=device)
    )

    cnn_model.to(device)
    cnn_model.eval()

    gradcam = GradCAM(cnn_model, cnn_model.features[-1])

    print("✅ Models loaded\n")

    return yolo_model, cnn_model, gradcam, yolo_class_names


# ================= PREDICT =================

def predict(image_path, yolo_model, cnn_model, gradcam, yolo_class_names):

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise ValueError("Image not found")

    results = yolo_model(img_bgr, conf=0.45, imgsz=800, verbose=False)[0]

    patch_predictions = []
    patch_count = 0
    yolo_detected_classes = []

    best_patch = None
    best_score = 0
    best_idx = 0

    for box in results.boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        cls_id = int(box.cls[0])
        if 0 <= cls_id < len(yolo_class_names):
            yolo_detected_classes.append(yolo_class_names[cls_id])

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = cnn_model(input_tensor)
            probs = F.softmax(output, dim=1).squeeze()

        pred_idx = torch.argmax(probs).item()
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = probs[pred_idx].item()

        # Save best patch for Grad-CAM
        if pred_conf > best_score:
            best_score = pred_conf
            best_patch = input_tensor.clone()
            best_idx = pred_idx

        # Histiocytoma boost
        if pred_class == "Histiocytoma" and pred_conf >= 0.5:
            pred_conf = min(pred_conf + 0.1, 1.0)

        # Filtering
        if pred_class == "Mast_cell_Tumor" and pred_conf >= 0.45:
            patch_predictions.append((pred_class, pred_conf))
            patch_count += 1

        elif pred_conf >= CONF_THRESHOLD:
            patch_predictions.append((pred_class, pred_conf))
            patch_count += 1

        # Draw box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_bgr, f"{pred_class} {pred_conf:.2f}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,0), 2)

    yolo_counter = Counter(yolo_detected_classes)

    # ===== CNN AGGREGATION =====

    cnn_prediction = "Negative or Inconclusive"
    cnn_confidence = 0.0

    if patch_count >= MIN_PATCHES:

        vote_counts = np.zeros(len(CLASS_NAMES), dtype=int)
        conf_sums = np.zeros(len(CLASS_NAMES), dtype=float)

        for cls, conf in patch_predictions:
            idx = CLASS_NAMES.index(cls)
            vote_counts[idx] += 1
            conf_sums[idx] += conf

        avg_conf = conf_sums / (vote_counts + 1e-6)
        lymphoma_ratio = vote_counts[CLASS_NAMES.index("Lymphoma")] / patch_count

        if lymphoma_ratio >= LYMPHOMA_THRESHOLD:
            cnn_prediction = "Lymphoma"
            cnn_confidence = lymphoma_ratio
        else:
            combined_score = vote_counts + avg_conf * 2.5
            top_idx = np.argmax(combined_score)
            cnn_prediction = CLASS_NAMES[top_idx]
            cnn_confidence = avg_conf[top_idx]

    # ===== YOLO OVERRIDE =====

    if yolo_counter.get("Histiocytoma", 0) >= 15:
        cnn_prediction = "Histiocytoma"
        cnn_confidence = max(cnn_confidence, 0.98)

    elif yolo_counter.get("Lymphoma", 0) >= YOLO_OVERRIDE_THRESHOLD:
        cnn_prediction = "Lymphoma"
        cnn_confidence = max(cnn_confidence, 0.98)

    elif yolo_counter.get("Mast cell", 0) >= 5:
        cnn_prediction = "Mast_cell_Tumor"
        cnn_confidence = max(cnn_confidence, 0.95)

    elif yolo_counter.get("TVT", 0) >= YOLO_OVERRIDE_THRESHOLD:
        cnn_prediction = "TVT"
        cnn_confidence = max(cnn_confidence, 0.98)

    # ===== GRAD-CAM =====

    heatmap_img = None

    if best_patch is not None:
        cam = gradcam.generate(best_patch, best_idx)
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

        original = cv2.resize(cv2.imread(image_path), (224,224))
        overlay = cv2.addWeighted(original, 0.6, cam, 0.4, 0)

        heatmap_img = overlay

    annotated_img = img_bgr

    # ===== OUTPUT =====

    print("\n===== RESULT =====")
    print(f"Prediction: {cnn_prediction}")
    print(f"Confidence: {cnn_confidence:.3f}")
    print(f"YOLO detections: {dict(yolo_counter)}")
    print(f"Patches used: {patch_count}")

    return annotated_img, heatmap_img, cnn_prediction, cnn_confidence


# ================= MAIN =================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)

    args = parser.parse_args()

    yolo_model, cnn_model, gradcam, yolo_names = load_models()

    predict(args.image, yolo_model, cnn_model, gradcam, yolo_names)
