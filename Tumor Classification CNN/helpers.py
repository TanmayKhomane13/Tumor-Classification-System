import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ==================== IMAGE PREPROCESSING ====================
def preprocess_xray_cv2(image_path, size = (128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    img = cv2.resize(img, size)
    img = img.astype('float32') / 255.0

    img = (img - 0.5) / 0.5

    img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    return img_tensor
# ==============================================================

# ==================== ACCURACY ====================
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            
            total += labels.size(0)
            correct += (preds == labels.view_as(preds)).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    return accuracy

# ==================== SHOW PREDICTIONS ON TEST SET ====================
def show_predictions(model, dataset, num_samples = 5):
    model.eval()
    plt.figure(num = "Testing set", figsize = (10, 2))

    for i in range(num_samples):
        image, label = dataset[i]
        input_img = image.unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_img)
            prob = torch.sigmoid(output)
            predicted = (prob >= 0.5).int().item()

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image.squeeze(), cmap = 'gray')
        plt.title(f"GT: {label} | Pred: {predicted}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==================== CUSTOM IMAGE PREDICTION ====================
def predict_custom_image(model, image_path):
    img_tensor = preprocess_xray_cv2(image_path)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)
        pred = (prob >= 0.5).int().item()
    
    plt.figure(num = "Your image")
    plt.imshow(img_tensor.squeeze(), cmap = 'gray')
    plt.title(f"Predicted Class: {'Malignant(1)' if pred else 'Benign(0)'}\nProb: {prob.item():.2f}")
    plt.axis('off')
    plt.show()

    return pred
