import os
import cv2
import random
from PIL import Image
from tqdm import tqdm
import albumentations as A


# ==============================
# CONFIGURATION
# ==============================

DATASET_ROOT = "cropped_dataset/train"

TARGET_COUNT = 2500

CLASSES = [
    "Histiocytoma",
    "Lymphoma",
    "Mast_cell",
    "TVT"
]


# ==============================
# SAFE AUGMENTATION PIPELINE
# ==============================

augmentor = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.4),
    A.Rotate(limit=3, border_mode=cv2.BORDER_REPLICATE, p=0.1),
])


# ==============================
# AUGMENTATION FUNCTION
# ==============================

def augment_class(class_name):

    input_folder = os.path.join(DATASET_ROOT, class_name)
    output_folder = os.path.join(DATASET_ROOT, class_name + "_aug")

    os.makedirs(output_folder, exist_ok=True)

    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    repeat_factor = TARGET_COUNT // len(image_files) + 1

    count = 0

    print(f"\nAugmenting class: {class_name}")

    for img_name in tqdm(image_files):

        img_path = os.path.join(input_folder, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for i in range(repeat_factor):

            augmented = augmentor(image=image)

            aug_img = Image.fromarray(augmented["image"])

            save_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"

            aug_img.save(os.path.join(output_folder, save_name))

            count += 1

            if count >= TARGET_COUNT:
                break

        if count >= TARGET_COUNT:
            break

    print(f"Generated {count} augmented images for {class_name}")

    move_augmented_images(class_name, output_folder)


# ==============================
# MOVE AUGMENTED IMAGES
# ==============================

def move_augmented_images(class_name, aug_folder):

    dest_folder = os.path.join(DATASET_ROOT, class_name)

    files = os.listdir(aug_folder)

    print(f"Moving augmented images for {class_name}...")

    for f in files:

        src = os.path.join(aug_folder, f)
        dst = os.path.join(dest_folder, f)

        os.rename(src, dst)

    os.rmdir(aug_folder)

    print(f"Completed merging augmented images for {class_name}")


# ==============================
# MAIN
# ==============================

def run_augmentation():

    for cls in CLASSES:

        augment_class(cls)

    print("\nAugmentation pipeline completed.")


if __name__ == "__main__":

    print("Starting dataset augmentation...")

    run_augmentation()
