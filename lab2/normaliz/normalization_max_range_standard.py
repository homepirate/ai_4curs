import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из файла
data = pd.read_csv('ex1data2.txt', header=None)
data.columns = ['Engine_Speed', 'Num_Gears', 'Price']

# Отображение первых нескольких записей

# Нормировка 1: делим на максимальное значение
data_normalized_1 = data.copy()
data_normalized_1.iloc[:, 0] = data.iloc[:, 0] / data.iloc[:, 0].max()  # Engine_Speed
data_normalized_1.iloc[:, 1] = data.iloc[:, 1] / data.iloc[:, 1].max()  # Num_Gears
data_normalized_1.iloc[:, 2] = data.iloc[:, 2] / data.iloc[:, 2].max()  # Price


# Нормировка 2: центрируем и делим на диапазон (max - min)
data_normalized_2 = data.copy()
data_normalized_2.iloc[:, 0] = (data.iloc[:, 0] - data.iloc[:, 0].mean()) / (data.iloc[:, 0].max() - data.iloc[:, 0].min())
data_normalized_2.iloc[:, 1] = (data.iloc[:, 1] - data.iloc[:, 1].mean()) / (data.iloc[:, 1].max() - data.iloc[:, 1].min())
data_normalized_2.iloc[:, 2] = (data.iloc[:, 2] - data.iloc[:, 2].mean()) / (data.iloc[:, 2].max() - data.iloc[:, 2].min())


# Нормировка 3: центрируем и делим на стандартное отклонение
data_normalized_3 = data.copy()
data_normalized_3.iloc[:, 0] = (data.iloc[:, 0] - data.iloc[:, 0].mean()) / data.iloc[:, 0].std()
data_normalized_3.iloc[:, 1] = (data.iloc[:, 1] - data.iloc[:, 1].mean()) / data.iloc[:, 1].std()
data_normalized_3.iloc[:, 2] = (data.iloc[:, 2] - data.iloc[:, 2].mean()) / data.iloc[:, 2].std()



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
plt.scatter(data_normalized_1['Engine_Speed'], data_normalized_1['Price'], color='green')
plt.title('Нормировка 1 (max)')
plt.xlabel('Скорость Оборота Двигателя (нормированная)')
plt.ylabel('Цена (нормированная)')

# Нормированные данные (второй способ)
plt.subplot(1, 4, 3)
plt.scatter(data_normalized_2['Engine_Speed'], data_normalized_2['Price'], color='red')
plt.title('Нормировка 2 (max - min)')
plt.xlabel('Скорость Оборота Двигателя (нормированная)')
plt.ylabel('Цена (нормированная)')

# Нормированные данные (третий способ)
plt.subplot(1, 4, 4)
plt.scatter(data_normalized_3['Engine_Speed'], data_normalized_3['Price'], color='purple')
plt.title('Нормировка 3 (стандартное отклонение)')
plt.xlabel('Скорость Оборота Двигателя (нормированная)')
plt.ylabel('Цена (нормированная)')

plt.tight_layout()
plt.savefig('normalization_max_range_standard.png')

# Стандартные функции Python
means_explicit = {}
std_devs_explicit = {}

for col in data.columns:
    means_explicit[col] = sum(data[col]) / len(data[col])
    std_devs_explicit[col] = (sum((data[col] - means_explicit[col]) ** 2) / (len(data[col]) - 1)) ** 0.5

print("Средние значения (явное вычисление):\n", means_explicit)
print("Стандартные отклонения (явное вычисление):\n", std_devs_explicit)

# Явно по определению
means_std = data.mean()
std_devs_std = data.std()

print("Средние значения (стандартные функции):\n", means_std)
print("Стандартные отклонения (стандартные функции):\n", std_devs_std)



