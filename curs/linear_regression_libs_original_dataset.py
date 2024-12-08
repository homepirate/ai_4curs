# optimized_ml_code.py
import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # For saving the model

# Load the dataset
data_path = '../data/csvdata.csv'
df = pd.read_csv(data_path)

# Drop unnecessary columns
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# Check and remove null values
df.dropna(inplace=True)

# Encode categorical variables
label_enc_city = LabelEncoder()
label_enc_loc = LabelEncoder()
df['City'] = label_enc_city.fit_transform(df['City'])
df['Location'] = label_enc_loc.fit_transform(df['Location'])

# Define features and target variable
X = df.drop(columns=['Price'])
y = df['Price']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Set up EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping])

# Evaluate the model
y_pred = model.predict(X_test).flatten()
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
r_squared = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Test MAE: {test_mae:.2f}")
print(f"R²: {r_squared:.2f}")
print(f"RMSE: {rmse:.2f}")

# Plot training history
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')

# Save the model and scaler
# model.save('regression_model.h5')
# joblib.dump(scaler, 'scaler.pkl')

# Example prediction
sample_input = np.array([[2, 645, 67, 1]])
sample_input_scaled = scaler.transform(sample_input)
predicted_price = model.predict(sample_input_scaled)
print(f"Predicted Price: {predicted_price[0][0]:.2f}")