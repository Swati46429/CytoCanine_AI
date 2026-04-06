import gradio as gr
import os
import cv2
import torch
import requests
import torch.nn.functional as F
import numpy as np
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
from collections import Counter

# ================= SETTINGS =================
cnn_model_path = "convnext_tiny_final_earlystop.pth"
yolo_path = "yolov8Vx_best.pt"

class_names = ['Histiocytoma', 'Lymphoma', 'Mast_cell_Tumor', 'Negative', 'TVT']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_PATCHES = 5
LYMPHOMA_THRESHOLD = 0.7
CONF_THRESHOLD = 0.6
YOLO_OVERRIDE_THRESHOLD = 3

# ================= GITHUB =================
GITHUB_REPO = "https://github.com/Swati46429/CytoCanine_AI"

# ================= BACKEND API =================
USAGE_API = "https://cytocanine-backend.onrender.com/usage"

def get_usage_count():
    try:
        r = requests.get(USAGE_API, timeout=5)
        data = r.json()
        return int(data["message"])   # ✅ backend sends "message"
    except Exception as e:
        print("Usage API error:", e)
        return 0

def update_usage():
    try:
        requests.post(USAGE_API, timeout=5)
    except:
        pass

# ================= STAR BUTTON =================
def open_github():
    return GITHUB_REPO
    
# ================= LOAD MODELS =================
print("Loading models...")

yolo_model = YOLO(yolo_path)
yolo_class_names = list(yolo_model.names.values())

cnn_model = models.convnext_tiny(weights=None)
num_features = cnn_model.classifier[2].in_features
cnn_model.classifier[2] = torch.nn.Linear(num_features, len(class_names))
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
cnn_model.to(device).eval()

print("YOLO + ConvNeXt loaded")
    
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

        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224,224),
                            mode="bilinear", align_corners=False)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

gradcam = GradCAM(cnn_model, cnn_model.features[-1])

# ================= UTIL =================
def load_usage():
    count = get_usage_count()
    if count == 0:
        return "📊 **No recorded usage yet**"

    return (
        "📊 **System Utilization Statistics accessed**\n"
        f"**{count} times** "
        "for cytological research and analysis."
    )
    
# ================= PREDICT =================
def predict(image):

    if image is None:
        return None,None,"No image","0 %",\
        load_usage(),""
        
    # ---------- UPDATE COUNTER ----------
    update_usage()
    count = get_usage_count()

    # ---------- image safety ----------
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = image.astype(np.uint8)

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ---------- YOLO ----------
    results = yolo_model(
        img_bgr, conf=0.45, imgsz=800, verbose=False
    )[0]

    patch_predictions=[]
    patch_count=0
    yolo_detected_classes=[]

    best_patch = None
    best_score = 0
    best_idx = 0

    for box in results.boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        crop = img_bgr[y1:y2,x1:x2]

        if crop.size==0:
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
        pred_class = class_names[pred_idx]
        pred_conf = probs[pred_idx].item()

        # best patch
        if pred_conf > best_score:
            best_score = pred_conf
            best_patch = input_tensor.clone()
            best_idx = pred_idx

        if pred_class=="Histiocytoma" and pred_conf>=0.5:
            pred_conf=min(pred_conf+0.1,1.0)

        if pred_class=="Mast_cell_Tumor" and pred_conf>=0.45:
            patch_predictions.append((pred_class,pred_conf))
            patch_count+=1
        elif pred_conf>=CONF_THRESHOLD:
            patch_predictions.append((pred_class,pred_conf))
            patch_count+=1

        cv2.rectangle(img_bgr,(x1,y1),(x2,y2),
                      (0,255,0),2)
        cv2.putText(img_bgr,
            f"{pred_class} {pred_conf:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,(0,255,0),2)

    yolo_counter = Counter(yolo_detected_classes)

    # ===== CNN AGGREGATION =====
    cnn_prediction = "Negative or Inconclusive"
    cnn_confidence = 0.0

    if patch_count >= MIN_PATCHES:
        vote_counts = np.zeros(len(class_names), dtype=int)
        conf_sums = np.zeros(len(class_names), dtype=float)

        for cls, conf in patch_predictions:
            idx = class_names.index(cls)
            vote_counts[idx] += 1
            conf_sums[idx] += conf

        avg_conf = conf_sums / (vote_counts + 1e-6)
        lymphoma_ratio = vote_counts[
            class_names.index("Lymphoma")
        ] / patch_count

        if lymphoma_ratio >= LYMPHOMA_THRESHOLD:
            cnn_prediction = "Lymphoma"
            cnn_confidence = lymphoma_ratio
        else:
            combined_score = vote_counts + avg_conf * 2.5
            top_idx = np.argmax(combined_score)
            cnn_prediction = class_names[top_idx]
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
    heatmap_img=None
    if best_patch is not None:
        cam = gradcam.generate(best_patch, best_idx)
        cam = np.uint8(255*cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        orig = cv2.resize(image,(224,224))
        overlay = cv2.addWeighted(orig,0.6,cam,0.4,0)
        heatmap_img = Image.fromarray(overlay)

        annotated = Image.fromarray(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    header_text = (
        "📊 **System Utilization Statistics accessed**\n"
        f"**{count} times** "
        "for cytological research and analysis."
    )
    textbox_text = f"📊 This app has been used: {count} times"

    return (
        annotated,
        heatmap_img,
        f"🐶 {cnn_prediction}",
        f"{round(float(cnn_confidence)*100,2)} %",
        header_text,      # <-- for TOP markdown
        textbox_text      # <-- for textbox
    )


# ================== GRADIO UI ==================
with gr.Blocks(css="""
img {max-height:350px !important;}
""") as demo:

    # TOP HEADER
    usage_header = gr.Markdown("⏳ **Fetching usage stats...**")

    gr.Markdown("""
### ⭐ Support this Research
**If this app helped your work, please star the repository**.
""")

    star_btn=gr.Button("⭐ Star on GitHub")
    
    star_btn.click(
    None,
    None,
    None,
    js=f"() => window.open('{GITHUB_REPO}', '_blank')"
)

    gr.Markdown("""
🔬 **AI-Assisted Cytological Diagnosis of Canine Round Cell Tumors**

🎯 **Website Purpose-CytoCanine AI**

This web-based application, **CytoCanine AI**, is designed to assist veterinary pathologists and researchers in the **automated detection and classification of canine round cell tumors** from cytological microscopic images

The system integrates a **YOLOv8x object detection model** to localize diagnostically relevant tumor regions, followed by a **ConvNeXt-Tiny deep learning classifier** to identify tumor subtypes

Additionally, **Explainable Artificial Intelligence (Explainable AI)** visualization, such as **Grad-CAM**, is incorporated to highlight important image regions influencing the model’s predictions, thereby improving transparency and clinical trust

This tool aims to: support **routine microscopic examination**, Reduce **diagnostic subjectivity**, Facilitate **research and educational analysis** in veterinary cytology

⚠️ **This tool is intended for research and educational use only and should not replace expert veterinary diagnosis**

🐶 **Pipeline-YOLOv8x detects diagnostically relevant regions → ConvNeXt-Tiny performs multi-class classification → Rule-based aggregation + Explainable AI (Grad-CAM)**

⚠️ **Upload Guidelines**

✔ **Only canine cytology microscope images (~100×)**  
✖ **No X-rays / mobile photos**

🔥 **Grad-CAM Color Guide**

🔴 **Red / Yellow → High importance (AI focused strongly)**  
🟢 **Green → Medium importance**  
🔵 **Blue → Low importance (AI ignored)**
""")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="📤 Upload Cytology Image",
                height=350
            )
            submit_btn = gr.Button("Submit")

        with gr.Column(scale=1):
            annotated_output = gr.Image(
                type="pil",
                label="📷 Annotated Output",
                height=350
            )
            gradcam_output = gr.Image(
                type="pil",
                label="🔥 Grad-CAM",
                height=350
            )
            prediction_text = gr.Textbox(
                label="🐶 Final Prediction")
            confidence_text = gr.Textbox(
                label="🔬 Confidence Score")
            usage_text = gr.Textbox(
                label="📊 App Usage Counter")

    submit_btn.click(
        fn=predict,
        inputs=input_image,
        outputs=[
            annotated_output,
            gradcam_output,
            prediction_text,
            confidence_text,
            usage_header, 
            usage_text
        ],
        api_name=False
    )

    # ===== expose API for README badge =====
    demo.load(
        fn=load_usage,
        inputs=None,
        outputs=usage_header
    )
    
demo.launch()
