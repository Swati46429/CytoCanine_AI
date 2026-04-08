import os
import cv2
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================

YOLO_DATASET_DIR = "dataset"
CNN_DATASET_DIR = "cropped_dataset"

RESIZE_DIM = 320
BOX_EXPAND_RATIO = 1.05

# ==============================
# CLASS MAPPING
# ==============================

class_id_to_name = {
    0: "Histiocytoma",
    1: "Lymphoma",
    2: "Mast_cell",
    3: "TVT"
}

# ==============================
# YOLO → BBOX CONVERSION
# ==============================

def yolo_to_bbox(box, img_w, img_h, expand_ratio=1.05):

    cx, cy, w, h = box

    w *= expand_ratio
    h *= expand_ratio

    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)

    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)

    return max(x1, 0), max(y1, 0), min(x2, img_w), min(y2, img_h)

# ==============================
# RESIZE WITH PADDING
# ==============================

def resize_with_padding(image, size=320):

    h, w = image.shape[:2]

    scale = size / max(h, w)

    resized = cv2.resize(image, (int(w * scale), int(h * scale)))

    h_res, w_res = resized.shape[:2]

    top = (size - h_res) // 2
    bottom = size - h_res - top

    left = (size - w_res) // 2
    right = size - w_res - left

    final = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_REFLECT
    )

    return final

# ==============================
# CREATE CROPPED DATASET
# ==============================

def create_cropped_dataset():

    summary = {}

    for split in ["train", "valid", "test"]:

        print(f"\nProcessing {split.upper()} dataset...")

        img_dir = os.path.join(YOLO_DATASET_DIR, split, "images")
        lbl_dir = os.path.join(YOLO_DATASET_DIR, split, "labels")

        output_root = os.path.join(CNN_DATASET_DIR, split)

        for class_name in class_id_to_name.values():
            os.makedirs(os.path.join(output_root, class_name), exist_ok=True)

        saved = 0
        skipped = 0

        for lbl_file in tqdm(os.listdir(lbl_dir)):

            if not lbl_file.endswith(".txt"):
                continue

            img_name = lbl_file.replace(".txt", ".jpg")

            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, lbl_file)

            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)

            if img is None:
                continue

            h, w = img.shape[:2]

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))
                box = list(map(float, parts[1:]))

                x1, y1, x2, y2 = yolo_to_bbox(box, w, h, BOX_EXPAND_RATIO)

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    skipped += 1
                    continue

                crop = resize_with_padding(crop, RESIZE_DIM)

                class_name = class_id_to_name[class_id]

                out_path = os.path.join(
                    output_root,
                    class_name,
                    f"{img_name.replace('.jpg','')}_{i}.jpg"
                )

                cv2.imwrite(out_path, crop)

                saved += 1

        summary[split] = {
            "saved": saved,
            "skipped": skipped
        }

    print("\nDataset Preparation Summary")

    for split, stats in summary.items():
        print(split.upper(), stats)

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    print("Starting dataset preparation...")

    create_cropped_dataset()

    print("\nCNN dataset preparation completed.")
