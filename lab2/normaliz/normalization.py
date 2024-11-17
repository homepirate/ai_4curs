import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('../ex1data2.txt', header=None)
data.columns = ['Engine_Speed', 'Num_Gears', 'Price']

# Отображение загруженных данных
# print(data.head())

# 1. Мин-Max нормализация
data_min_max = (data - data.min()) / (data.max() - data.min())

# 2. Стандартная нормализация
data_standard = (data - data.mean()) / data.std()

# 3. Нормализация по длине вектора
data_normalized_length = data / np.linalg.norm(data, axis=0)

# # Отображение нормализованных данных
# print("Мин-Max Нормализация:\n", data_min_max)
# print("Стандартная Нормализация:\n", data_standard)
# print("Нормализация по длине вектора:\n", data_normalized_length)

plt.figure(figsize=(15, 10))

# Первый график
plt.subplot(2, 2, 1)
plt.scatter(data['Engine_Speed'], data['Price'], color='blue')
plt.title('Исходные Признаки')
plt.xlabel('Скорость Оборота Двигателя')
plt.ylabel('Цена')

# Второй график (Мин-Max Нормализация)
plt.subplot(2, 2, 2)
plt.scatter(data_min_max['Engine_Speed'], data_min_max['Price'], color='green')
plt.title('Мин-Max Нормализация')
plt.xlabel('Скорость Оборота Двигателя')
plt.ylabel('Цена')

# Третий график (Стандартная Нормализация)
plt.subplot(2, 2, 3)
plt.scatter(data_standard['Engine_Speed'], data_standard['Price'], color='red')
plt.title('Стандартная Нормализация')
plt.xlabel('Скорость Оборота Двигателя')
plt.ylabel('Цена')

# Четвертый график
plt.subplot(2, 2, 4)
plt.scatter(data_standard['Engine_Speed'], data_normalized_length['Price'], color='purple')
plt.title('Стандартная Нормализация с Длиной')
plt.xlabel('Скорость Оборота Двигателя')
plt.ylabel('Цена')

plt.tight_layout()
plt.savefig('normalization.png')

# Вычисление среднего значения
means = data.mean()
# Вычисление стандартного отклонения
std_devs = data.std()

print("Средние значения:\n", means)
print("Стандартные отклонения:\n", std_devs)


