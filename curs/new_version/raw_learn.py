import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data_path = '../../data/csvdata2.csv'
data = pd.read_csv(data_path)


# 1. Function to normalize data and encode categorical features
def normalize_data(data):
    # Encoding categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Normalizing numerical data
    scaler = MinMaxScaler()
    data[data.columns] = scaler.fit_transform(data[data.columns])

    return data, label_encoders, scaler


# 2. Function to shuffle dataset
def shuffle_dataset(data):
    return data.sample(frac=1).reset_index(drop=True)


# 3. Function to split dataset
def split_dataset(data, target_column, test_size=0.2, val_size=0.1):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Further split train+val into train and validation sets
    val_relative_size = val_size / (1 - test_size)  # Adjust validation size relative to the train+val set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_relative_size,
                                                      random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


# Applying the functions
# 1. Normalize the dataset
normalized_data, encoders, scaler = normalize_data(data)

# 2. Shuffle the dataset
shuffled_data = shuffle_dataset(normalized_data)

# 3. Split the dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(shuffled_data, target_column='Price')

# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on validation data
y_val_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")
