import pickle
import numpy as np

# Load the model and scaler
with open('tumor_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

w_out = model_data['weights']
b_out = model_data['bias']
scaler = model_data['scaler']

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(x, w, b):
    z = np.dot(x, w) + b
    return sigmoid(z)

# user input
size = float(input("Enter tumor size: "))
age = float(input("Enter age: "))
density = float(input("Enter tumor density: "))
user = np.array([size, age, density])
user_scaled = scaler.transform(user.reshape(1, -1))

# ------------------------------------ predict class ------------------------------------------------
z_user = np.dot(user, w_out) + b_out
probability = sigmoid(z_user)

# Tumor Size
if user[0] < 1:
    print("Microtumor\n -> may be pre-malignant")
elif user[0] >=1 and user[0] <= 2:
    print("Small Tumor\n -> potentially resectable with minimal impact")
elif user[0] >= 2 and user[0] <= 5:
    print("Medium Tumor\n -> intermediate risk")
elif user[0] >= 5 and user[0] <= 10:
    print("Large Tumor\n -> increased mass effect")
else:
    print("Giant Tumor\n -> can cause organ displacement")

# Patient Age - Based
if user[1] >= 1 and user[1] <= 18:
    print("Pediatric Tumor\n Common Types: Neuroblastoma, Wilms Tumor, Retinoblastoma, Medulloblastoma\n  -> Often genetic, congenital, or embryonic in origin")
elif user[1] >= 19 and user[1] <= 40:
    print("Young Adult Tumor\n Common Types: Testicular cancer, Hodgkin's lymphoma\n  -> Aggressive but often highly treatable")
elif user[1] >= 41 and user[1] <= 60:
    print("Middle-Aged Tumor\n Common Types: Breast cancer, Colorectal cancer, Lung cancer\n  -> Linked to environmental factors (eg. smoking, diet)")
else:
    print("Elderly Tumor\n Common Types: Prostate cancer, Pancreatic cancer, Slow growing brain tumors\n  -> Often slower-growing but harder to treat due to comorbidities")

# Tumor Density - Based
if user[2] >= -100 and user[2] <= 200:
    print("Hypodense Tumor")
elif user[2] >= 20 and user[2] <= 60:
    print("Isodense Tumor")
elif user[2] >= 60 and user[2] <= 300:
    print("Hyperdense Tumor")
else:
    print("Mixed Density Tumor")

predictedClass = 1 if probability >= 0.5 else 0
print(f"Predicted Probability: {probability[0]}")
if probability < 0.2:
    advice = "No immediate risk. Follow regular check-ups."
elif probability < 0.5:
    advice = "Further testing recommended (MRI, Biopsy)."
else:
    advice = "Consult an oncologist immediately!"

print(f"Predicted class: {'Malignant (1)' if predictedClass == 1 else 'Benign (0)'}")
print(f"Medical Advice: {advice}")
