import os

DATASET_DIR = "dataset"

TRAIN_IMAGES = os.path.join(DATASET_DIR, "train/images")
TRAIN_LABELS = os.path.join(DATASET_DIR, "train/labels")

VAL_IMAGES = os.path.join(DATASET_DIR, "valid/images")
VAL_LABELS = os.path.join(DATASET_DIR, "valid/labels")


def check_dataset():

    print("Checking dataset structure...")

    required_dirs = [
        TRAIN_IMAGES,
        TRAIN_LABELS,
        VAL_IMAGES,
        VAL_LABELS
    ]

    for d in required_dirs:

        if not os.path.exists(d):
            print(f"Missing directory: {d}")
        else:
            print(f"Found: {d}")


if __name__ == "__main__":
    check_dataset()
