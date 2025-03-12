import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'ex2data1.txt'  # Specify your file path
data = pd.read_csv(file_path, header=None, names=["Vibration", "Rotation", "Label"])

# Separate the features (X) and labels (y)
X_original = data[["Vibration", "Rotation"]].values
y = data["Label"].values

# Feature scaling
X_mean = np.mean(X_original, axis=0)
X_std = np.std(X_original, axis=0)
X = (X_original - X_mean) / X_std

# Add intercept term to X
X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add a column of ones for the intercept term

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient descent
def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    cost_history = []

    for _ in range(num_iters):
        gradient = (1/m) * (X.T @ (sigmoid(X @ theta) - y))
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history

# Initialize parameters
m, n = X.shape
initial_theta = np.zeros(n)
alpha = 0.1  # Increased learning rate
num_iters = 5000  # Increased number of iterations

# Train the model
final_theta, cost_history = gradient_descent(X, y, initial_theta, alpha, num_iters)

# Predictions
def predict(X, theta):
    probabilities = sigmoid(X @ theta)
    return (probabilities >= 0.5).astype(int)

y_pred = predict(X, final_theta)
accuracy = np.mean(y_pred == y)

print(f"Accuracy: {accuracy:.2f}")

# Visualize decision boundary
plt.figure(figsize=(8, 6))

# Use the original scale of the data for plotting
x_min, x_max = data["Vibration"].min() - 1, data["Vibration"].max() + 1
y_min, y_max = data["Rotation"].min() - 1, data["Rotation"].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Scale the meshgrid before prediction
X_grid = np.c_[np.ones((xx.size, 1)),
               (xx.ravel() - X_mean[0]) / X_std[0],
               (yy.ravel() - X_mean[1]) / X_std[1]]

Z = predict(X_grid, final_theta)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

# Plot the original data points (unscaled)
for label, color, marker in zip([0, 1], ['red', 'blue'], ['o', 'x']):
    subset = data[data["Label"] == label]
    plt.scatter(subset["Vibration"], subset["Rotation"],
                label=f"Class {label}", c=color, marker=marker, edgecolors='k')

plt.xlabel("Vibration")
plt.ylabel("Rotation")
plt.legend()
plt.title("Decision Boundary")
plt.savefig("manual_decision_boundary.png")

# Plot the cost history
plt.figure(figsize=(8, 6))
plt.plot(range(len(cost_history)), cost_history, label="Cost")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Convergence")
plt.legend()
plt.savefig("cost_convergence.png")
