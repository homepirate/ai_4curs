import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 21)
y = x + np.random.uniform(-2, 2, x.shape)

theta1_values = np.linspace(0, 2, 5)

# Построение графиков для разных значений theta1
plt.figure(figsize=(10, 6))
for theta1 in theta1_values:
    h_x = theta1 * x  # Аппроксимирующая прямая
    plt.plot(x, h_x, color='blue', alpha=0.25)

# Отображение зашумлённых данных
plt.scatter(x, y, color='red', label='Noisy Data')

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
plt.savefig('2-1.png')


theta1_values = np.linspace(0, 2, 200)

J_theta1 = [(np.sum(((theta1 * x) - y) ** 2)) for theta1 in theta1_values]

min_index = np.argmin(J_theta1)
theta1_min = theta1_values[min_index]

plt.figure()
# Построение графика ошибки J в зависимости от theta1
plt.plot(theta1_values, J_theta1, label='Ошибка J(theta1)')
# plt.scatter(theta1_min, J_theta1[min_index], color='red', label=f'Минимум J ({theta1_min:.2f})')
print(f'Минимум theta1 ({theta1_min})')

plt.title('Функционал ошибки  (для зашумленных данных)')
plt.xlabel('theta1')
plt.ylabel('J(theta1)')
plt.grid(True)
plt.savefig('2-2.png')

