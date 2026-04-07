from ultralytics import YOLO
import torch


def train_model():

    print("CUDA available:", torch.cuda.is_available())

    model = YOLO("yolov8x.pt")

    model.train(

        data="dataset/data.yaml",

        epochs=300,
        imgsz=800,
        batch=8,

        optimizer="AdamW",
        lr0=0.0005,
        cos_lr=True,

        dropout=0.1,
        patience=70,

        mosaic=1.0,
        mixup=0.3,

        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,

        degrees=0.5,
        translate=0.1,
        scale=0.5,
        shear=0.2,
        perspective=0.0005,

        name="tumor_detection_yolov8x",

        save=True,
        verbose=True
    )


if __name__ == "__main__":
    train_model()
