# from_scratch_regression.py
import numpy as np
import pandas as pd

# Load the dataset
data_path = '../data/csvdata2.csv'
df = pd.read_csv(data_path)

# Encode categorical variables manually
df['City'] = df['City'].astype('category').cat.codes
df['Location'] = df['Location'].astype('category').cat.codes

# Define features and target
X = df.drop(columns=['Price']).values
y = df['Price'].values.reshape(-1, 1)

# Min-max normalization
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_scaled = (X - X_min) / (X_max - X_min)

# Add bias term to X
X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

# Train-test split
split_idx = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Initialize weights
np.random.seed(42)
weights = np.random.randn(X_train.shape[1], 1)

# Hyperparameters
learning_rate = 0.01
epochs = 100000
m = len(X_train)

# Gradient descent
for epoch in range(epochs):
    predictions = X_train @ weights
    errors = predictions - y_train
    gradient = (1 / m) * (X_train.T @ errors)
    weights -= learning_rate * gradient

    if epoch % 100 == 0:
        loss = (1 / (2 * m)) * np.sum(errors ** 2)
        print(f"Epoch {epoch}, Loss: {loss:.2f}")

# Predictions on test set
y_pred = X_test @ weights

# Metrics
r_squared = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) **2)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print(f"R²: {r_squared:.2f}")
print(f"RMSE: {rmse:.2f}")

# Example prediction
sample_input = np.array([[2, 645, 67, 1]])
sample_input_scaled = (sample_input - X_min) / (X_max - X_min)
sample_input_scaled = np.hstack([np.ones((sample_input_scaled.shape[0], 1)), sample_input_scaled])
predicted_price = sample_input_scaled @ weights
print(f"Predicted Price: {predicted_price[0][0]:.2f}")