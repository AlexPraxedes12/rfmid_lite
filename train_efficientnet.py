import os
import torch
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
import numpy as np
import torch.nn as nn
import torch.optim as optim

class RFMiDDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """Dataset loader for the RFMiD CSV using 28 abbreviated disease labels."""

        # Columns that correspond to the disease labels to predict
        self.disease_cols = [
            "DR",
            "ARMD",
            "MH",
            "DN",
            "MYA",
            "BRVO",
            "TSLN",
            "ERM",
            "LS",
            "MS",
            "CSR",
            "ODC",
            "CRVO",
            "TV",
            "AH",
            "ODP",
            "ODE",
            "ST",
            "AION",
            "PT",
            "RT",
            "RS",
            "CRS",
            "EDN",
            "RPEC",
            "MHL",
            "RP",
            "OTHER",
            "Normal"
        ]

        # Load the label CSV
        df = pd.read_csv(csv_file)

        # Drop the Disease_Risk column if present
        df = df.drop(columns=["Disease_Risk"], errors="ignore")

        # Ensure that all expected columns are present
        missing = set(["ID"] + self.disease_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns {missing} in {csv_file}")

        # Keep only ID and disease columns
        self.labels = df[["ID"] + self.disease_cols].copy()

        # Convert disease labels to float32 for the model
        self.labels[self.disease_cols] = self.labels[self.disease_cols].astype("float32")

        # Print sanity check on the resulting label tensor shape
        print(f"Loaded labels with shape {self.labels[self.disease_cols].shape}")

        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Ensure image names read from the CSV are converted to strings when
        # constructing the image path. This avoids type errors if the ID column
        # contains numeric values.
        img_name = os.path.join(self.img_dir, str(self.labels.iloc[idx, 0]) + ".png")
        image = Image.open(img_name).convert('RGB')

        # Extract the disease labels for this image
        label_values = self.labels.loc[idx, self.disease_cols].values
        label = torch.tensor(label_values, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate_model(loader, model, device, save_path="/content/drive/MyDrive/results_efficientnet.txt"):
    """Evaluate model on a given loader and return metrics.

    Predictions are obtained by applying a sigmoid activation followed by
    binarization using a 0.5 threshold. The function prints a classification
    report and additional metrics, then writes them to ``save_path``.
    """

    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    report = classification_report(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    try:
        auc_macro = roc_auc_score(y_true, y_prob, average="macro")
    except Exception:
        auc_macro = "N/A"

    # Display the evaluation metrics without emojis to avoid encoding issues
    try:
        print("Classification Report:\n", report)
    except UnicodeEncodeError:
        # Fall back to printing a safely encoded version
        safe_report = report.encode("utf-8", errors="ignore").decode("utf-8")
        print("Classification Report:\n", safe_report)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Macro: {f1_macro:.4f}")
    print(f"F1 Micro: {f1_micro:.4f}")
    print(f"Macro AUC: {auc_macro}")

    safe_report = report.encode("utf-8", errors="ignore").decode("utf-8")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=== EfficientNet Evaluation Report ===\n\n")
        f.write(safe_report + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n")
        f.write(f"F1 Micro: {f1_micro:.4f}\n")
        f.write(f"AUC Macro: {auc_macro}\n")

    print(f"Results saved to: {save_path}")
    return acc, f1_macro, f1_micro, report

def train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=5):
    model.train()
    best_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f'[Epoch {epoch+1}] Validation Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "efficientnet_rfmid_best.pth")
            # Inform the user when the best model is updated without using emojis
            print("Saved best model")

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths (ajusta según tu Drive o entorno local)
    train_dir = "/content/Training_Set/Training"
    train_csv = "/content/Training_Set/RFMiD_Training_Labels.csv"
    val_dir = "/content/Evaluation_Set/Validation"
    val_csv = "/content/Evaluation_Set/RFMiD_Validation_Labels.csv"
    test_dir = "/content/Test_Set/Test"
    test_csv = "/content/Test_Set/RFMiD_Testing_Labels.csv"

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Loaders
    train_set = RFMiDDataset(train_csv, train_dir, transform)
    val_set = RFMiDDataset(val_csv, val_dir, transform)
    test_set = RFMiDDataset(test_csv, test_dir, transform)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    # Model
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_set.disease_cols))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    model = train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=10)

    # Test
    print("\n Testing results:")
    print(evaluate_model(test_loader, model, device))

if __name__ == "__main__":
    main()
