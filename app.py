import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import numpy as np
import cv2
import os
from huggingface_hub import hf_hub_download

# --- SETTINGS ---
class_names = ['Histiocytoma', 'Lymphoma', 'Mast_cell', 'Negative', 'TVT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Download models from Hugging Face dataset if not exists ---
if not os.path.exists("yolov8Vx_best.pt"):
    yolo_model_path = hf_hub_download(
        repo_id="DeepBioSwati/cytocanine_models",
        filename="yolov8Vx_best.pt",
        repo_type="dataset"
    )
else:
    yolo_model_path = "yolov8Vx_best.pt"

if not os.path.exists("efficientnet_final_earlystop.pth"):
    cnn_model_path = hf_hub_download(
        repo_id="DeepBioSwati/cytocanine_models",
        filename="efficientnet_final_earlystop.pth",
        repo_type="dataset"
    )
else:
    cnn_model_path = "efficientnet_final_earlystop.pth"

# --- MODEL PARAMETERS ---
min_required_patches = 5
lymphoma_threshold = 0.7
yolo_override_threshold = 5
cnn_conf_threshold = 0.50

# --- LOAD MODELS ---
@st.cache_resource(show_spinner=False)
def load_models():
    # Fix for PyTorch 2.8 unpickling YOLO model
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])

    # YOLO
    yolo_model = YOLO(yolo_model_path)

    # EfficientNet
    cnn_model = EfficientNet.from_name("efficientnet-b2")
    cnn_model._fc = torch.nn.Linear(cnn_model._fc.in_features, len(class_names))
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn_model.eval().to(device)

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return yolo_model, cnn_model, transform

yolo_model, cnn_model, transform = load_models()

# --- PREDICTION FUNCTION ---
def predict(image):
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = yolo_model(img_bgr)[0]
    yolo_class_counts = [0] * len(class_names)
    patch_predictions = []
    patch_count = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        yolo_cls_idx = int(box.cls[0])
        yolo_class_counts[yolo_cls_idx] += 1

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = cnn_model(input_tensor)
            probs = F.softmax(output, dim=1).squeeze()

        pred_idx = torch.argmax(probs).item()
        pred_class = class_names[pred_idx]
        pred_conf = probs[pred_idx].item()
        patch_predictions.append((pred_class, pred_conf))
        patch_count += 1

        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, f"{pred_class} {pred_conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # CNN aggregation
    if patch_count < min_required_patches:
        cnn_prediction = "Negative or Inconclusive (Insufficient evidence)"
    else:
        total_probs = np.zeros(len(class_names))
        vote_count_arr = np.zeros(len(class_names), dtype=int)

        for cls, conf in patch_predictions:
            idx = class_names.index(cls)
            if cls == "TVT" and conf >= 0.35:
                vote_count_arr[idx] += 1
                total_probs[idx] += conf
            elif conf >= cnn_conf_threshold:
                vote_count_arr[idx] += 1
                total_probs[idx] += conf

        avg_probs = total_probs / patch_count
        lymphoma_ratio = vote_count_arr[class_names.index("Lymphoma")] / patch_count
        if lymphoma_ratio >= lymphoma_threshold:
            cnn_prediction = "Lymphoma (by 70% rule)"
        else:
            top_idx = np.argmax(avg_probs)
            if avg_probs[top_idx] >= cnn_conf_threshold or class_names[top_idx] == "TVT":
                cnn_prediction = class_names[top_idx]
            else:
                cnn_prediction = "Low confidence"

    yolo_counts_dict = {cls: count for cls, count in zip(class_names, yolo_class_counts)}
    yolo_candidates = {cls: cnt for cls, cnt in yolo_counts_dict.items() if cls != "Negative"}

    if yolo_candidates:
        if "TVT" in yolo_candidates and yolo_candidates["TVT"] >= 3:
            final_prediction = "TVT"
        else:
            top_yolo_class = max(yolo_candidates, key=yolo_candidates.get)
            if yolo_candidates[top_yolo_class] >= yolo_override_threshold:
                final_prediction = top_yolo_class
            else:
                final_prediction = cnn_prediction
    else:
        final_prediction = cnn_prediction

    annotated_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    patch_json = {f"Patch {i+1}": {"Predicted_Class": cls, "Confidence": float(conf)}
                  for i, (cls, conf) in enumerate(patch_predictions)}
    return annotated_image, final_prediction, patch_json

# --- STREAMLIT UI ---
st.title("🐶 CytoCanine AI: YOLO + EfficientNet Tumor Detection & Classification")
st.write("Upload a cytology image → YOLO detects patches → EfficientNet classifies → CNN rules + YOLO override applied → Final result.")

uploaded_file = st.file_uploader("Upload Cytology Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        annotated_img, final_prediction, patch_json = predict(image)

    st.image(annotated_img, caption="Annotated Output", use_column_width=True)
    st.subheader("Final Prediction")
    st.success(final_prediction)

    st.subheader("Patch-wise Predictions")
    st.json(patch_json)
