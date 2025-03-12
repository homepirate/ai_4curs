import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
x = np.arange(1, 21)
y = x  # y_i = x_i

# Значения theta1
theta1_values = np.linspace(0, 2, 5)

# Построение графиков для разных значений theta1
plt.figure(figsize=(10, 6))
for theta1 in theta1_values:
    h_x = theta1 * x  # Аппроксимирующая прямая
    plt.plot(x, h_x, color='blue', alpha=0.25)

# Отображение исходных данных
plt.scatter(x, y, color='red', label='Experimental Data')

# Настройки осей
plt.xlim(0, 21)
plt.xticks(np.arange(0, 22, 1))
plt.ylim(0, max(y) * max(theta1_values) + 1)

# Настройки графика
plt.title("Аппроксимирующая прямая")
plt.xlabel("x")
plt.ylabel("h(x) = theta1 * x")
plt.grid(True)
plt.legend()
plt.savefig('1-1.png')


theta1_values = np.linspace(0, 2, 200)

# График функционала ошибки
J_values = []
for theta1 in theta1_values:
    h_x = theta1 * x
    J = np.sum((h_x - y) ** 2)
    J_values.append(J)

plt.figure()
plt.plot(theta1_values, J_values, label='J(θ₁)', color='b')
plt.xlabel('theta1')
plt.ylabel('J(theta1)')
plt.title('Функционал ошибки (без шума)')
plt.legend()
plt.grid(True)
plt.savefig('1-2.png')

theta1_min = theta1_values[np.argmin(J_values)]
print(f'Минимальное theta1: {theta1_min}')
