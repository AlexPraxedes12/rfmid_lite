import os
import torch
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import classification_report
import torch.nn as nn
import torch.optim as optim

class RFMiDDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """Dataset that loads only the seven disease labels from the RFMiD CSV."""

        # Columns that correspond to the disease labels we care about
        self.disease_cols = [
            "Diabetic Retinopathy",
            "Hypertensive Retinopathy",
            "Glaucoma",
            "Cataract",
            "Age-related Macular Degeneration",
            "Retinal Detachment",
            "Others",
        ]

        # Load the label CSV
        df = pd.read_csv(csv_file)

        # Ensure that all expected columns are present
        missing = set(self.disease_cols + ["ID"]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns {missing} in {csv_file}")

        # Keep only ID and disease columns, discarding metadata and symptom columns
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

        # Extract the seven disease labels for this image
        label_values = self.labels.loc[idx, self.disease_cols].values
        label = torch.tensor(label_values, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate_model(loader, model, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
    return classification_report(all_labels, all_preds, zero_division=0, output_dict=False)

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
            print("✔️ Saved best model")

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
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    model = train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=10)

    # Test
    print("\n📊 Testing results:")
    print(evaluate_model(test_loader, model, device))

if __name__ == "__main__":
    main()
