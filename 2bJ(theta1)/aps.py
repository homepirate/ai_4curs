import numpy as np
import matplotlib.pyplot as plt
from ferror import theta1_min_clean, theta1_min_noisy

# Исходные данные
x = np.arange(1, 21)
y_clean = x  # Чистые данные
y_noisy = x + np.random.uniform(-2, 2, x.shape)  # Шумные данные

# Значения theta1
theta1_values = np.linspace(0, 2, 5)

# Создание фигуры для обоих графиков
plt.figure(figsize=(10, 6))

# Построение аппроксимирующих прямых для чистых данных
for i, theta1 in enumerate(theta1_values):
    if i == 0:
        h_x = theta1_min_clean * x  # Аппроксимирующая прямая
    else:
        h_x = theta1 * x
    plt.plot(x, h_x, color='red', alpha=0.25, linewidth=2, label='Чистые данные' if i == 0 else "")


# Построение аппроксимирующих прямых для защумленных данных
for i, theta1 in enumerate(theta1_values):
    if i == 0:
        h_x_noisy = theta1_min_noisy * x  # Аппроксимирующая прямая
    else:
        h_x_noisy = theta1 * x  # Аппроксимирующая прямая для шумных данных
    plt.plot(x, h_x_noisy, color='blue', alpha=0.25, linestyle='--', label='Зашумленные данные'
    if theta1 == theta1_values[0] else "")

# Отображение чистых данных
plt.scatter(x, y_clean, color='green', label='Чистые данные', marker='o')

# Отображение зашумлённых данных
plt.scatter(x, y_noisy, color='red', label='Зашумленные данные', marker='x')

# Настройки осей
plt.xlim(0, 21)
plt.xticks(np.arange(0, 22, 1))
plt.ylim(0, max(y_noisy) * max(theta1_values) + 1)

# Настройки графика
plt.title("Аппроксимирующая прямая для чистых и шумных данных")
plt.xlabel("x")
plt.ylabel("h(x) = theta1 * x")
plt.grid(True)
plt.legend()
plt.savefig('combined.png')  # Сохранение объединенного графика
