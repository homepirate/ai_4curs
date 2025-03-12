import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize_max(data):
    """Нормировка на основе максимального значения."""
    max_values = [max(column) for column in data.T]
    normalized_data = np.array([[value / max_value for value, max_value in zip(row, max_values)] for row in data])
    return normalized_data


def normalize_min_max(data):
    """Центрирование и нормировка на основе диапазона (max - min)."""
    means = [sum(column) / len(column) for column in data.T]
    min_values = [min(column) for column in data.T]
    max_values = [max(column) for column in data.T]

    normalized_data = np.array([
        [(value - mean) / (max_value - min_value) for value, mean, max_value, min_value in
         zip(row, means, max_values, min_values)]
        for row in data
    ])
    return normalized_data


def normalize_z_score(data):
    """Центрирование и нормировка на основе стандартного отклонения."""
    means = [sum(column) / len(column) for column in data.T]
    std_devs = [(sum((value - mean) ** 2 for value in column) / (len(column) - 1)) ** 0.5 for mean, column in
                zip(means, data.T)]

    normalized_data = np.array([
        [(value - mean) / std_dev for value, mean, std_dev in zip(row, means, std_devs)]
        for row in data
    ])
    return normalized_data


# Загрузка данных из файла
data = pd.read_csv('ex1data2.txt', header=None)
data.columns = ['Engine_Speed', 'Num_Gears', 'Price']
data_values = data.values  # Преобразование данных в массив NumPy

# Нормировка данных
data_normalized_1 = normalize_max(data_values)
data_normalized_2 = normalize_min_max(data_values)
data_normalized_3 = normalize_z_score(data_values)

# Визуализация исходных и нормализованных данных
plt.figure(figsize=(20, 6))

# Исходные данные
plt.subplot(1, 4, 1)
plt.scatter(data['Engine_Speed'], data['Price'], color='blue')
plt.title('Исходные Признаки')
plt.xlabel('Скорость Оборота Двигателя')
plt.ylabel('Цена')

# Нормированные данные (первый способ)
plt.subplot(1, 4, 2)
plt.scatter(data_normalized_1[:, 0], data_normalized_1[:, 2], color='green')
plt.title('Нормировка 1 (max)')
plt.xlabel('Скорость Оборота Двигателя (нормированная)')
plt.ylabel('Цена (нормированная)')

# Нормированные данные (второй способ)
plt.subplot(1, 4, 3)
plt.scatter(data_normalized_2[:, 0], data_normalized_2[:, 2], color='red')
plt.title('Нормировка 2 (max - min)')
plt.xlabel('Скорость Оборота Двигателя (нормированная)')
plt.ylabel('Цена (нормированная)')

# Нормированные данные (третий способ)
plt.subplot(1, 4, 4)
plt.scatter(data_normalized_3[:, 0], data_normalized_3[:, 2], color='purple')
plt.title('Нормировка 3 (стандартное отклонение)')
plt.xlabel('Скорость Оборота Двигателя (нормированная)')
plt.ylabel('Цена (нормированная)')

plt.tight_layout()
plt.savefig('normalization_max_range_standard_manual.png')
