import os
import cv2
import uuid
import albumentations as A
from tqdm import tqdm


augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.HueSaturationValue(p=0.4),
    A.GaussNoise(p=0.3),
    A.MotionBlur(p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.Resize(640, 640)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


class_multipliers = {
    0: 4,
    1: 0,
    2: 1,
    3: 2
}


def run_augmentation(train_images, train_labels, aug_images, aug_labels):

    os.makedirs(aug_images, exist_ok=True)
    os.makedirs(aug_labels, exist_ok=True)

    image_files = [f for f in os.listdir(train_images)
                   if f.endswith(".jpg") or f.endswith(".png")]

    for img_file in tqdm(image_files):

        img_path = os.path.join(train_images, img_file)
        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(train_labels, label_file)

        if not os.path.exists(label_path):
            continue

        image = cv2.imread(img_path)

        bboxes = []
        class_labels = []

        with open(label_path) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            bbox = list(map(float, parts[1:]))

            bboxes.append(bbox)
            class_labels.append(cls_id)

        main_cls = class_labels[0]
        aug_times = class_multipliers.get(main_cls, 0)

        for _ in range(aug_times):

            augmented = augmentor(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )

            unique = uuid.uuid4().hex[:6]

            new_img = f"{img_file.split('.')[0]}_aug_{unique}.jpg"

            cv2.imwrite(
                os.path.join(aug_images, new_img),
                augmented["image"]
            )
