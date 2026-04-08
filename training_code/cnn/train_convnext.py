import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
import os

from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight


# ==============================
# CONFIGURATION
# ==============================

DATASET_DIR = "cropped_dataset"

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 5
INPUT_SIZE = 224

MODEL_SAVE_PATH = "models/convnext_tiny_final.pth"


# ==============================
# DEVICE
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================
# DATA TRANSFORMS
# ==============================

train_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=(-2, 2)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ==============================
# DATASETS
# ==============================

image_datasets = {

    "train": datasets.ImageFolder(
        os.path.join(DATASET_DIR, "train"),
        transform=train_transform
    ),

    "valid": datasets.ImageFolder(
        os.path.join(DATASET_DIR, "valid"),
        transform=val_transform
    )

}


dataloaders = {

    x: DataLoader(
        image_datasets[x],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    for x in ["train", "valid"]
}


class_names = image_datasets["train"].classes
num_classes = len(class_names)

print("Detected classes:", class_names)


# ==============================
# CLASS WEIGHTS
# ==============================

labels = [label for _, label in image_datasets["train"]]

class_weights = compute_class_weight(
    "balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)


# ==============================
# MODEL
# ==============================

weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1

model = models.convnext_tiny(weights=weights)

in_features = model.classifier[2].in_features

model.classifier[2] = nn.Linear(in_features, num_classes)

# Freeze backbone

for param in model.features.parameters():
    param.requires_grad = False

# Train classifier head

for param in model.classifier.parameters():
    param.requires_grad = True


model = model.to(device)


# ==============================
# LOSS & OPTIMIZER
# ==============================

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    patience=3,
    factor=0.5
)


# ==============================
# TRAINING LOOP
# ==============================

def train_model():

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    patience_counter = 0

    since = time.time()

    for epoch in range(NUM_EPOCHS):

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        for phase in ["train", "valid"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "valid":

                scheduler.step(epoch_acc)

                if epoch_acc > best_acc:

                    best_acc = epoch_acc
                    best_weights = copy.deepcopy(model.state_dict())

                    patience_counter = 0

                    print("Validation improved. Saving best model.")

                else:

                    patience_counter += 1

                    print(f"No improvement ({patience_counter}/{PATIENCE})")

                    if patience_counter >= PATIENCE:

                        print("\nEarly stopping triggered.")
                        model.load_state_dict(best_weights)

                        return model

    time_elapsed = time.time() - since

    print(
        f"\nTraining completed in {time_elapsed//60:.0f}m "
        f"{time_elapsed%60:.0f}s"
    )

    model.load_state_dict(best_weights)

    return model


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":

    print("Starting ConvNeXt training...")

    model = train_model()

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"\nModel saved to {MODEL_SAVE_PATH}")
