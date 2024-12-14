import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = np.loadtxt("ex2data1.txt", delimiter=",")
X = data[:, :2]  # Признаки: вибрация и вращение
y = data[:, 2]   # Метки: исправность (0) или неисправность (1)

# Добавление полиномиальных признаков
def add_polynomial_features(X, degree=2):
    m, n = X.shape
    result = [np.ones(m)]  # Добавляем столбец единиц (смещение)
    for i in range(1, degree + 1):
        for j in range(i + 1):
            result.append((X[:, 0] ** (i - j)) * (X[:, 1] ** j))
    return np.column_stack(result)

X_poly = add_polynomial_features(X, degree=2)

# Инициализация параметров
theta = np.zeros(X_poly.shape[1])
alpha = 0.01  # Скорость обучения
iterations = 1000

# Функция сигмоиды
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Функция стоимости
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return -(1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))

# Градиентный спуск
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        theta -= (alpha / m) * (X.T @ (h - y))
    return theta

# Обучение модели
theta = gradient_descent(X_poly, y, theta, alpha, iterations)

# Предсказание
def predict(X, theta):
    return (sigmoid(X @ theta) >= 0.5).astype(int)

# Оценка модели
y_pred = predict(X_poly, theta)
accuracy = np.mean(y_pred == y) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Визуализация границы принятия решений
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Применение модели для сетки
grid = add_polynomial_features(np.c_[xx.ravel(), yy.ravel()], degree=2)
Z = predict(grid, theta).reshape(xx.shape)

# Построение графика
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Class 0 (Functional)', marker='o')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Class 1 (Faulty)', marker='x')
plt.xlabel("Vibration")
plt.ylabel("Rotation")
plt.title("Decision Boundary")
plt.legend()
plt.savefig("2.png")
