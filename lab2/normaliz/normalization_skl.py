import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Reading data from file
df = pd.read_csv('../ex1data2.txt', header=None, names=['engine_rpm', 'gear_count', 'price'])

# Calculating mean and std manually
mean_engine_rpm = df['engine_rpm'].mean()
std_engine_rpm = np.sqrt(np.mean((df['engine_rpm'] - mean_engine_rpm)**2))

mean_gear_count = df['gear_count'].mean()
std_gear_count = np.sqrt(np.mean((df['gear_count'] - mean_gear_count)**2))

mean_price = df['price'].mean()
std_price = np.sqrt(np.mean((df['price'] - mean_price)**2))

print(f'Mean values:\nEngine RPM: {mean_engine_rpm:.2f}\nGear Count: {mean_gear_count:.2f}\nPrice: {mean_price:.2f}')
print(f'\nStd Deviation:\nEngine RPM: {std_engine_rpm:.2f}\nGear Count: {std_gear_count:.2f}\nPrice: {std_price:.2f}')

# Feature scaling methods
minmax_scaler = MinMaxScaler()
minmax_scaled_data = minmax_scaler.fit_transform(df)
minmax_df = pd.DataFrame(minmax_scaled_data, columns=df.columns)

standard_scaler = StandardScaler()
standard_scaled_data = standard_scaler.fit_transform(df)
standard_df = pd.DataFrame(standard_scaled_data, columns=df.columns)

robust_scaler = RobustScaler()
robust_scaled_data = robust_scaler.fit_transform(df)
robust_df = pd.DataFrame(robust_scaled_data, columns=df.columns)

# Visualizing original and scaled features
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 15), sharex=True)

axes[0].scatter(range(len(df)), df['engine_rpm'], label='Original Data')
axes[0].set_title('Engine RPM')
axes[0].legend()

axes[1].scatter(range(len(minmax_df)), minmax_df['engine_rpm'], color='g', label='MinMax Normalized')
axes[1].set_title('Min-Max Normalization of Engine RPM')
axes[1].legend()

axes[2].scatter(range(len(standard_df)), standard_df['engine_rpm'], color='b', label='Standard Scaled')
axes[2].set_title('Standard Scaling of Engine RPM')
axes[2].legend()

axes[3].scatter(range(len(robust_df)), robust_df['engine_rpm'], color='r', label='Robust Scaled')
axes[3].set_title('Robust Scaling of Engine RPM')
axes[3].legend()

plt.savefig("normalization_skl.png")
