import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # For saving models

# Load the dataset
data_path = '../../data/csvdata2.csv'
data = pd.read_csv(data_path)

# 1. Function to normalize data and encode categorical features
def normalize_data(data):
    # Identify categorical and numerical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_cols.remove('Price')  # Exclude target variable

    # Encoding categorical features
    label_encoders = {}
    for column in categorical_cols:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Normalizing numerical data
    scaler = MinMaxScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    return data, label_encoders, scaler, categorical_cols, numerical_cols

# 2. Function to shuffle dataset
def shuffle_dataset(data):
    return data.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Function to split dataset
def split_dataset(data, target_column, test_size=0.2, val_size=0.1):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Further split train+val into train and validation sets
    val_relative_size = val_size / (1 - test_size)  # Adjust validation size relative to the train+val set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

# Applying the functions
# 1. Normalize the dataset
normalized_data, encoders, scaler, categorical_cols, numerical_cols = normalize_data(data)

# 2. Shuffle the dataset
shuffled_data = shuffle_dataset(normalized_data)

# 3. Split the dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    shuffled_data, target_column='Price'
)

# Save feature names
feature_names = X_train.columns.tolist()
joblib.dump(feature_names, 'feature_names.joblib')

# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on validation data
y_val_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)
print(f'Validation MSE: {mse}')
print(f'Validation R²: {r2}')

# Save the model, encoders, scaler, and feature names
joblib.dump(model, 'linear_regression_model.joblib')
joblib.dump(encoders, 'label_encoders.joblib')
joblib.dump(scaler, 'minmax_scaler.joblib')

print("Model and preprocessing objects have been saved successfully.")
