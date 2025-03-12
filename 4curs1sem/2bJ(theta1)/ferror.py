import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 21)
y_clean = x  # Чистые данные
y_noisy = x + np.random.uniform(-2, 2, x.shape)  # Шумные данные

theta1_values = np.linspace(0, 2, 200)

# Функционал ошибки для чистых данных
J_clean_values = []
for theta1 in theta1_values:
    h_x = theta1 * x
    J_clean = np.sum((h_x - y_clean) ** 2)
    J_clean_values.append(J_clean)

# Функционал ошибки для шумных данных
J_noisy_values = []
for theta1 in theta1_values:
    h_x = theta1 * x
    J_noisy = np.sum((h_x - y_noisy) ** 2)
    J_noisy_values.append(J_noisy)

plt.figure(figsize=(10, 6))

plt.plot(theta1_values, J_clean_values, label='Ошибка J (чистые данные)', color='blue')


plt.plot(theta1_values, J_noisy_values, label='Ошибка J (зашумленные данные)', color='orange')

min_index_noisy = np.argmin(J_noisy_values)
theta1_min_noisy = theta1_values[min_index_noisy]

# Настройки графика
plt.title('Функционал ошибки для чистых и шумных данных')
plt.xlabel('theta1')
plt.ylabel('J(theta1)')
plt.xlim(0.5, 1.5)
plt.ylim(-5, 200)

plt.grid(True)
plt.legend()
plt.savefig('combined_error_function.png')  # Сохранение объединенного графика

min_index_clean = np.argmin(J_clean_values)
theta1_min_clean = theta1_values[min_index_clean]

print(f'Минимальное theta1 для чистых данных: {theta1_min_clean}')
print(f'Минимальное theta1 для шумных данных: {theta1_min_noisy}')
