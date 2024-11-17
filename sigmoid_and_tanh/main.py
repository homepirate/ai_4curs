import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def derivative_tanh(x):
    t = np.tanh(x)
    return 1 - t**2

# Вычисление значений сигмоиды
points = [0, 3, -3, 8, -8, 15, -15]
sigmoid_values = [sigmoid(x) for x in points]

print("Значения функции сигмоида: ")
for i, value in enumerate(sigmoid_values):
    print(f"σ({points[i]}) = {value:.15f}")

# График сигмоиды
x = np.linspace(-20, 20, num=1000)
y = sigmoid(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label="Sigmoid")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.title("График функции сигмоидов")
plt.legend()
plt.grid(True)
plt.savefig('sigmoid.png')

# Гиперболические функции
x = np.linspace(-20, 20, num=1000)

sinh_x = np.sinh(x)
cosh_x = np.cosh(x)
tanh_x = np.tanh(x)

plt.figure(figsize=(10, 6))
plt.plot(x, sinh_x, label="sinh(x)")
plt.plot(x, cosh_x, label="cosh(x)")
plt.plot(x, tanh_x, label="tanh(x)")
plt.xlabel("x")
plt.ylabel("Hyperbolic Functions")
plt.title("Графики гиперболических функций")
plt.legend()
plt.grid(True)
plt.savefig('giperb.png')

# Производные
derivative_sigmoid_values = [derivative_sigmoid(p) for p in points]
derivative_tanh_values = [derivative_tanh(p) for p in points]

print("\nПроизводные сигмоиды: ")
for i, value in enumerate(derivative_sigmoid_values):
    print(f"σ'({points[i]}) = {value:.15f}")

print("\nПроизводные гиперболического тангеса: ")
for j, value in enumerate(derivative_tanh_values):
    print(f"tanh'({points[j]}) = {value:.15f}")
