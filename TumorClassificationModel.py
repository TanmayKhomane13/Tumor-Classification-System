import numpy as np
import matplotlib.pyplot as plt
import math, copy
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def computeGradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(x[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * x[i, j]
        dj_db += err_i
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db

def gradientDescent(x, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = computeGradient(x, y, w, b)

        # update the parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

    return w, b, J_history


# load dataset
df = pd.read_csv("tumorData.csv")

# Training data (now with 3 features)
x_train = df[["Tumor Size (cm)", "Patient Age", "Tumor Density (HU)"]].values
y_train = df[["Malignant (1) or Benign (0)"]].values

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # normalize features

# initial settings
w_init = np.zeros_like(x_train[0])
b_init = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradientDescent(x_train, y_train, w_init, b_init, alph, iters)

model_data = {
    'weights': w_out,
    'bias': b_out,
    'scaler': scaler
}

with open('tumor_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# Predict on training data
# z_train = np.dot(x_train, w_out) + b_out
# y_pred = (sigmoid(z_train) >= 0.5).astype(int)  # Convert probabilities to 0 or 1

# # --------- user input -------------
# os.system('clear')
# # Calculate accuracy
# accuracy = accuracy_score(y_train, y_pred) * 100
# print(f"Model Accuracy: {accuracy:.2f}%")

# size = float(input("Enter tumor size: "))
# age = float(input("Enter age: "))
# density = float(input("Enter tumor density: "))
# user = np.array([size, age, density])
# user_scaled = scaler.transform(user.reshape(1, -1))

# # ------------------------------------ predict class ------------------------------------------------
# z_user = np.dot(user, w_out) + b_out
# probability = sigmoid(z_user)

# # Tumor Size
# if user[0] < 1:
#     print("Microtumor\n -> may be pre-malignant")
# elif user[0] >=1 and user[0] <= 2:
#     print("Small Tumor\n -> potentially resectable with minimal impact")
# elif user[0] >= 2 and user[0] <= 5:
#     print("Medium Tumor\n -> intermediate risk")
# elif user[0] >= 5 and user[0] <= 10:
#     print("Large Tumor\n -> increased mass effect")
# else:
#     print("Giant Tumor\n -> can cause organ displacement")

# # Patient Age - Based
# if user[1] >= 1 and user[1] <= 18:
#     print("Pediatric Tumor\n Common Types: Neuroblastoma, Wilms Tumor, Retinoblastoma, Medulloblastoma\n  -> Often genetic, congenital, or embryonic in origin")
# elif user[1] >= 19 and user[1] <= 40:
#     print("Young Adult Tumor\n Common Types: Testicular cancer, Hodgkin's lymphoma\n  -> Aggressive but often highly treatable")
# elif user[1] >= 41 and user[1] <= 60:
#     print("Middle-Aged Tumor\n Common Types: Breast cancer, Colorectal cancer, Lung cancer\n  -> Linked to environmental factors (eg. smoking, diet)")
# else:
#     print("Elderly Tumor\n Common Types: Prostate cancer, Pancreatic cancer, Slow growing brain tumors\n  -> Often slower-growing but harder to treat due to comorbidities")

# # Tumor Density - Based
# if user[2] >= -100 and user[2] <= 200:
#     print("Hypodense Tumor")
# elif user[2] >= 20 and user[2] <= 60:
#     print("Isodense Tumor")
# elif user[2] >= 60 and user[2] <= 300:
#     print("Hyperdense Tumor")
# else:
#     print("Mixed Density Tumor")

# predictedClass = 1 if probability >= 0.5 else 0
# print(f"Predicted Probability: {probability[0]}")
# if probability < 0.2:
#     advice = "No immediate risk. Follow regular check-ups."
# elif probability < 0.5:
#     advice = "Further testing recommended (MRI, Biopsy)."
# else:
#     advice = "Consult an oncologist immediately!"

# print(f"Predicted class: {'Malignant (1)' if predictedClass == 1 else 'Benign (0)'}")
# print(f"Medical Advice: {advice}")



