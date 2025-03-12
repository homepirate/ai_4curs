"""
Решение задачи:
1) Вычисление значений функции сигмоида в заданных точках и построение её графика.
2) Построение трёх графиков гиперболических функций: sinh(x), cosh(x), tanh(x).
3) Вычисление производных функции сигмоида и гиперболического тангенса, а также их графиков.
"""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative_np(x):
    """Производная функции сигмоида."""
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative_np(x):
    """Производная функции гиперболического тангенса."""
    t = np.tanh(x)
    return 1 - t**2


# Заданные точки
points = [0, 3, -3, 8, -8, 15, -15]

print("Вычисление значений функции сигмоида в заданных точках с 15 знаками после запятой:")
for x in points:
    y = sigmoid(x)
    print(f"y({x}) = {y:.15f}")

# Построение графика функции сигмоида
x_vals = np.linspace(-20, 20, 400)
y_vals = sigmoid(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label='Сигмоида', color='blue')
plt.scatter(points, [sigmoid(x) for x in points], color='red', zorder=5, label='Заданные точки')
for point in points:
    plt.annotate(f'y({point})={sigmoid(point):.4f}', (point, sigmoid(point)),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
plt.title('График функции сигмоида')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('sigmoid_plot.png')  # Сохранение графика сигмоида

# 2. Построение трёх графиков гиперболических функций на одной картинке с использованием plt.subplot

# Диапазон значений x для гиперболических функций
x_hyper = np.linspace(-5, 5, 400)

# Вычисление значений гиперболических функций
sinh_vals = np.sinh(x_hyper)
cosh_vals = np.cosh(x_hyper)
tanh_vals = np.tanh(x_hyper)

# Создание подграфиков
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 2.1. График sinh(x)
axs[0].plot(x_hyper, sinh_vals, label='sinh(x)', color='green', linewidth=2)
axs[0].set_title('График функции sinh(x)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('sinh(x)')
axs[0].grid(True)
axs[0].legend()

# 2.2. График cosh(x)
axs[1].plot(x_hyper, cosh_vals, label='cosh(x)', color='orange', linewidth=2)
axs[1].set_title('График функции cosh(x)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('cosh(x)')
axs[1].grid(True)
axs[1].legend()

# 2.3. График tanh(x)
axs[2].plot(x_hyper, tanh_vals, label='tanh(x)', color='purple', linewidth=2)
axs[2].set_title('График функции tanh(x)')
axs[2].set_xlabel('x')
axs[2].set_ylabel('tanh(x)')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.savefig('hyperbolic_functions_subplot.png')  # Сохранение графиков гиперболических функций

# 3. Вычисление производных сигмоида и гиперболического тангенса и их графики

# Вычисление производных для заданных точек (опционально)
print("\nВычисление производных функций в заданных точках:")
for x in points:
    sig_der = sigmoid_derivative_np(x)
    tanh_der = 1 - np.tanh(x)**2
    print(f"y'(сигмоида)({x}) = {sig_der:.15f}, y'(tanh)({x}) = {tanh_der:.15f}")

# Диапазон значений x для производных
x_der = np.linspace(-5, 5, 400)

# Вычисление значений производных
sig_der_vals = sigmoid_derivative_np(x_der)
tanh_der_vals = tanh_derivative_np(x_der)

plt.figure(figsize=(10, 6))
plt.plot(x_der, sig_der_vals, label="y'(сигмоида)", color='red', linewidth=2)
plt.plot(x_der, tanh_der_vals, label="y'(tanh(x))", color='blue', linewidth=2)
plt.title('Графики производных функций сигмоида и tanh(x)')
plt.xlabel('x')
plt.ylabel('Значение производной')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('derivatives_plot.png')  # Сохранение графика производных
