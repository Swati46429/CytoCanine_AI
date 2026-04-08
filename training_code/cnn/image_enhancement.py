import os
import cv2
import numpy as np
from tqdm import tqdm


# ==============================
# CONFIGURATION
# ==============================

DATASET_ROOT = "cropped_dataset"

CLASSES = [
    "Histiocytoma",
    "Lymphoma",
    "Mast_cell",
    "TVT"
]

RESIZE_DIM = 320


# ==============================
# RESIZE WITH PADDING
# ==============================

def resize_with_padding(img, size=320):

    h, w = img.shape[:2]

    scale = size / max(h, w)

    resized = cv2.resize(img, (int(w * scale), int(h * scale)))

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
# GENTLE SHARPENING
# ==============================

def gentle_enhance(img):

    kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3, -0.5],
        [0, -0.5, 0]
    ])

    return cv2.filter2D(img, -1, kernel)


# ==============================
# FEATHER LIGHT ENHANCEMENT
# ==============================

def feather_light_enhance(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=0.5,
        tileGridSize=(6, 6)
    )

    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))

    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    kernel = np.array([
        [0, -0.03, 0],
        [-0.03, 1.1, -0.03],
        [0, -0.03, 0]
    ])

    final = cv2.filter2D(enhanced, -1, kernel)

    return final


# ==============================
# MAIN ENHANCEMENT PIPELINE
# ==============================

def enhance_dataset():

    enhanced_count = 0

    for split in ["train", "valid", "test"]:

        for cls in CLASSES:

            folder = os.path.join(DATASET_ROOT, split, cls)

            if not os.path.exists(folder):
                continue

            print(f"\nEnhancing {split}/{cls}")

            for fname in tqdm(os.listdir(folder)):

                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                path = os.path.join(folder, fname)

                img = cv2.imread(path)

                if img is None:
                    continue

                if img.shape[:2] != (RESIZE_DIM, RESIZE_DIM):
                    img = resize_with_padding(img, RESIZE_DIM)

                if cls in ["Mast_cell", "Lymphoma"]:
                    img_enhanced = feather_light_enhance(img)
                else:
                    img_enhanced = gentle_enhance(img)

                cv2.imwrite(path, img_enhanced)

                enhanced_count += 1

    print("\nEnhancement completed.")
    print(f"Total enhanced images: {enhanced_count}")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    print("Starting image enhancement pipeline...")

    enhance_dataset()

    print("\nImage enhancement finished.")
