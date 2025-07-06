import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model import CNN
from helpers import calculate_accuracy

# ========== Hyperparameters ==========
BATCH_SIZE = 32
IMAGE_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.0005

# ========== Data Transforms ==========
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ========== Datasets ==========
train_dataset = ImageFolder(root = 'data/train', transform = transform)
test_dataset = ImageFolder(root = 'data/test', transform = transform)

train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

# class imbalance checker
# import os

# train_benign = len(os.listdir("data/train/benign"))
# train_malignant = len(os.listdir("data/train/malignant"))
# test_benign = len(os.listdir("data/test/benign"))
# test_malignant = len(os.listdir("data/test/malignant"))

# print(f"Train Benign: {train_benign}")
# print(f"Train Malignant: {train_malignant}")
# print(f"Test Benign: {test_benign}")
# print(f"Test Malignant: {test_malignant}")
# # ===================================


# ========== Model ==========
model = CNN()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)


# ========== Training =================
print("Starting Training.......\n")

for epoch in range(EPOCHS):
    model.train()
    COST = 0

    for images, labels in train_loader:
        labels = labels.unsqueeze(1).float()

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        COST += loss.item()
    
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {COST:.4f}")
    calculate_accuracy(model, test_loader)

# ========== Save Model ==========
torch.save(model.state_dict(), "tumor_cnn.pth")
print("\nModel saved as 'tumor_cnn.pth'")
