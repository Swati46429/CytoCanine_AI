import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ConvNeXt_Tiny_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model():

    print("Loading trained ConvNeXt-Tiny model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    model_path = "models/convnext_tiny_final_earlystop.pth"
    test_dir = "dataset/test"

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    class_names = test_dataset.classes
    num_classes = len(class_names)

    print(f"Detected classes: {class_names}")

    # Load model
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)

    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Classification report
    print("\nClassification Report:\n")

    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    plt.title("Confusion Matrix - ConvNeXt Tiny")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

    # Per class accuracy
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for i in range(len(y_true)):

        label = y_true[i]
        pred = y_pred[i]

        if label == pred:
            class_correct[label] += 1

        class_total[label] += 1

    print("\nPer-Class Accuracy:\n")

    for i in range(num_classes):

        acc = 100 * class_correct[i] / class_total[i]

        print(f"{class_names[i]:20s}: {acc:.2f}%")

    # Overall accuracy
    overall_acc = 100 * np.sum(
        np.array(y_true) == np.array(y_pred)
    ) / len(y_true)

    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")



if __name__ == "__main__":

    evaluate_model()
