import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CNN
from helpers import show_predictions, predict_custom_image

# ======= Parameters =======
IMAGE_SIZE = 128
MODEL_PATH = 'tumor_cnn.pth'
NUM_SAMPLES = 5

# ======= Transform =======
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ======= Load Test Dataset =======
test_dataset = ImageFolder(root = 'data/test', transform = transform)

# ======= Load Model =======
model = CNN()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ======= Show Predictions =======
# show_predictions(model, test_dataset, num_samples = NUM_SAMPLES)

image_path = '/Users/tanmaykhomane/Desktop/Tumor Classification CNN/Te-noTr_0001.jpg'

# Predict
predict_custom_image(model, image_path)
