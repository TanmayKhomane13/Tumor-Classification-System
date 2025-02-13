import numpy as np
import matplotlib.pyplot as plt
import math, copy

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def computeCost(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(x[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost /= m
    return cost

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

        if i < 100000:
            J_history.append(computeCost(x, y, w, b))
        
    return w, b, J_history


# Training data (now with 3 features)
x_train = np.array([
    [0.5, 1.5, 0.8], [1, 1, 1.2], [1.5, 0.5, 0.9], [3, 0.5, 1.8], [2, 2, 2.5], [1, 2.5, 1.0],
    [1.0, 1.0, 1.1], [1.2, 1.3, 1.4], [1.8, 0.8, 1.0], [2.1, 1.7, 1.7], [2.8, 1.2, 1.9],
    [3.0, 1.5, 2.0], [1.7, 2.2, 1.8], [2.3, 2.0, 2.2], [2.7, 2.4, 2.5], [3.5, 2.0, 2.8],
    [3.0, 2.5, 2.6], [1.6, 1.8, 1.7], [2.6, 2.1, 2.4], [2.9, 1.6, 2.3], [3.2, 1.9, 2.7],
    [1.9, 1.0, 1.5], [2.0, 2.5, 2.0], [2.4, 1.8, 2.3], [2.7, 2.2, 2.6], [3.1, 2.3, 2.9],
])

y_train = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1])

# initial settings
w_init = np.zeros_like(x_train[0])
b_init = 0.
alph = 0.1
iters = 10000

w_out, b_out, _ = gradientDescent(x_train, y_train, w_init, b_init, alph, iters)

# --------- user input -------------
size = float(input("Enter tumor size: "))
age = float(input("Enter age: "))
density = float(input("Enter tumor density: "))
user = np.array([size, age, density])

# predict class
z_user = np.dot(user, w_out) + b_out
probability = sigmoid(z_user)
predictedClass = 1 if probability >= 0.5 else 0
print(f"Predicted Probability: {probability}")
if probability < 0.2:
    advice = "No immediate risk. Follow regular check-ups."
elif probability < 0.5:
    advice = "Further testing recommended (MRI, Biopsy)."
else:
    advice = "Consult an oncologist immediately!"

print(f"Predicted class: {'Malignant (1)' if predictedClass == 1 else 'Benign (0)'}")
print(f"Medical Advice: {advice}")

# -------------------------- plotting ---------------------------------------------------------
# 3D Plotting
fig = plt.figure(figsize=(7, 6), label = "Tumor Classification")
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of data
ax.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], x_train[y_train == 0, 2],
           c='g', marker='o', label='Benign')
ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], x_train[y_train == 1, 2],
           c='r', marker='x', label='Malignant')

# User input point
ax.scatter(size, age, density, c='b', marker='s', label='User Input')

# Decision boundary
xx, yy = np.meshgrid(np.linspace(0, 4, 10), np.linspace(0, 4, 10))
zz = (-b_out - w_out[0] * xx - w_out[1] * yy) / w_out[2]
ax.plot_surface(xx, yy, zz, color='blue', alpha=0.3)

ax.set_xlabel("Tumor Size")
ax.set_ylabel("Age")
ax.set_zlabel("Tumor Density")
plt.legend()
plt.show()
