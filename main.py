
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import timm
from torch.optim import Adam

# Configuración
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 28
CSV_PATH = "rfmid_labels.csv"
IMAGE_DIR = "rfmid_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset personalizado
class RFMiDDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["ID"] + ".jpg"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx].drop("ID").values.astype("float32")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

    def __len__(self):
        return len(self.data)

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Cargar datos
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)

train_dataset = RFMiDDataset("train.csv", IMAGE_DIR, transform=transform)
val_dataset = RFMiDDataset("val.csv", IMAGE_DIR, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Modelo
model = timm.create_model("efficientnet_b0", pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Entrenamiento
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Guardar modelo
torch.save(model.state_dict(), "efficientnet_rfmid.pth")
