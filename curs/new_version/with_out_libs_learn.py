import pandas as pd
import numpy as np
import pickle  # For saving models
import sys
import matplotlib.pyplot as plt  # For plotting training loss

# 1. Load the dataset
data_path = '../../data/csvdata.csv'
data = pd.read_csv(data_path)


# 2. Function to encode categorical features and normalize numerical features
def preprocess_data(data, target_column):
    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)  # Exclude target variable

    # One-Hot Encoding for categorical features
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    # Update categorical_cols after one-hot encoding
    # No need for label_encoders as one-hot encoding is used

    # Normalizing numerical features using Min-Max Scaling
    scalers = {}
    for col in numerical_cols:
        min_val = data[col].min()
        max_val = data[col].max()
        if max_val - min_val != 0:
            data[col] = (data[col] - min_val) / (max_val - min_val)
        else:
            data[col] = 0.0  # If all values are the same
        scalers[col] = {'min': min_val, 'max': max_val}

    # Normalize the target variable
    target_min = data[target_column].min()
    target_max = data[target_column].max()
    if target_max - target_min != 0:
        data[target_column] = (data[target_column] - target_min) / (target_max - target_min)
    else:
        data[target_column] = 0.0  # If all target values are the same
    target_scaler = {'min': target_min, 'max': target_max}

    return data, scalers, target_scaler


# 3. Function to shuffle the dataset
def shuffle_dataset(data, random_seed=42):
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(data))
    return data.iloc[shuffled_indices].reset_index(drop=True)


# 4. Function to split the dataset into train, validation, and test sets
def split_dataset(data, target_column, test_size=0.2, val_size=0.1):
    total_size = len(data)
    test_end = int(total_size * test_size)
    val_end = test_end + int(total_size * val_size)

    test_set = data.iloc[:test_end]
    val_set = data.iloc[test_end:val_end]
    train_set = data.iloc[val_end:]

    X_train = train_set.drop(columns=[target_column]).values.astype(np.float64)
    y_train = train_set[target_column].values.reshape(-1, 1).astype(np.float64)

    X_val = val_set.drop(columns=[target_column]).values.astype(np.float64)
    y_val = val_set[target_column].values.reshape(-1, 1).astype(np.float64)

    X_test = test_set.drop(columns=[target_column]).values.astype(np.float64)
    y_test = test_set[target_column].values.reshape(-1, 1).astype(np.float64)

    return X_train, X_val, X_test, y_train, y_val, y_test


# 5. Custom Linear Regression Implementation with Enhanced Features
class LinearRegressionCustom:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate=0.01, epochs=1000, batch_size=32):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        # Number of batches
        n_batches = int(np.ceil(n_samples / batch_size))

        # To store loss values for plotting
        losses = []

        for epoch in range(1, epochs + 1):
            # Shuffle the data at the beginning of each epoch
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Predictions
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # Compute gradients
                dw = (2 / X_batch.shape[0]) * np.dot(X_batch.T, (y_pred - y_batch))
                db = (2 / X_batch.shape[0]) * np.sum(y_pred - y_batch)

                # Update parameters
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db

                # Check for numerical issues
                if np.isnan(self.weights).any() or np.isnan(self.bias):
                    print("NaN detected in weights or bias. Stopping training.")
                    sys.exit(1)
                if np.isinf(self.weights).any() or np.isinf(self.bias):
                    print("Infinite value detected in weights or bias. Stopping training.")
                    sys.exit(1)

            # Compute and store loss every epoch
            y_pred_epoch = np.dot(X, self.weights) + self.bias
            loss = np.mean((y_pred_epoch - y) ** 2)
            losses.append(loss)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# 6. Evaluation Metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score_custom(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


# 7. Applying the preprocessing functions
# Preprocess the data
normalized_data, scalers, target_scaler = preprocess_data(data, target_column='Price')

# Shuffle the dataset
shuffled_data = shuffle_dataset(normalized_data)

# Split the dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    shuffled_data, target_column='Price', test_size=0.2, val_size=0.1
)


# Check for NaN or infinite values in splits
def check_data_sanity(X, y, split_name):
    if np.isnan(X).any() or np.isnan(y).any():
        print(f"NaN detected in {split_name} set.")
    if np.isinf(X).any() or np.isinf(y).any():
        print(f"Infinite values detected in {split_name} set.")


check_data_sanity(X_train, y_train, "training")
check_data_sanity(X_val, y_val, "validation")
check_data_sanity(X_test, y_test, "test")

# Save feature names for future reference
feature_names = list(normalized_data.drop(columns=['Price']).columns)
# with open('feature_names.pkl', 'wb') as f:
#     pickle.dump(feature_names, f)

# 8. Initialize and train the custom linear regression model
model = LinearRegressionCustom()
model.fit(X_train, y_train, learning_rate=0.01, epochs=1000, batch_size=32)

# 9. Make predictions on the validation set
y_val_pred = model.predict(X_val)


# Inverse transform the predictions and true values to original scale
def inverse_transform(y, scaler):
    return y * (scaler['max'] - scaler['min']) + scaler['min']


y_val_original = inverse_transform(y_val, target_scaler)
y_val_pred_original = inverse_transform(y_val_pred, target_scaler)

# 10. Evaluate the model
mse = mean_squared_error(y_val_original, y_val_pred_original)
r2 = r2_score_custom(y_val_original, y_val_pred_original)
print(f'Validation MSE: {mse}')
print(f'Validation R²: {r2}')

# 11. Save the model and preprocessing objects
# Save the trained model parameters
# with open('linear_regression_model.pkl', 'wb') as f:
#     pickle.dump({'weights': model.weights, 'bias': model.bias}, f)
#
# # Save the scalers
# with open('minmax_scaler.pkl', 'wb') as f:
#     pickle.dump(scalers, f)
#
# # Save the target scaler
# with open('target_scaler.pkl', 'wb') as f:
#     pickle.dump(target_scaler, f)

print("Model and preprocessing objects have been saved successfully.")
