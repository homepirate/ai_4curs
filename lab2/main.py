import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filename):
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, :-1].values  # Признаки
    y = data.iloc[:, -1].values  # Целевая переменная
    return X, y


def featureNormalize(X):
    mu = np.mean(X, axis=0)  # Среднее значение
    sigma = np.std(X, axis=0)  # Стандартное отклонение
    X_norm = (X - mu) / sigma  # Нормировка
    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    m = len(y)  # количество примеров
    J = (1 / (2 * m)) * np.sum(np.square(X @ theta - y))
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta = theta - (alpha / m) * (X.T @ (X @ theta - y))
        J_history[i] = computeCostMulti(X, y, theta)

    return theta, J_history


def normalEqn(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


# Загрузка данных
X, y = load_data('ex1data2.txt')
m = len(y)

# Нормировка признаков
X_norm, mu, sigma = featureNormalize(X)

# Добавление единичного столбца к X
X_norm = np.hstack((np.ones((m, 1)), X_norm))

# Установка параметров
alpha = 0.01
num_iters = 500
theta = np.zeros(X_norm.shape[1])

# Запуск градиентного спуска
theta, J_history = gradientDescentMulti(X_norm, y, theta, alpha, num_iters)

# Визуализация стоимости
plt.plot(range(1, num_iters + 1), J_history, '-b', linewidth=2)
plt.xlabel('Количество итераций')
plt.ylabel('Стоимость J')
plt.title('Сходимость градиентного спуска')
plt.savefig('1.png')

# Предсказание стоимости трактора с использованием градиентного спуска
engine_speed = (2104 - mu[0]) / sigma[0]  # Нормировка (для градиентного спуска)
num_gears = (3 - mu[1]) / sigma[1]  # Нормировка (для градиентного спуска)

predicted_price = np.array([[1, engine_speed, num_gears]]) @ theta
print(f'Предсказанная стоимость трактора (градиентный спуск): {predicted_price[0]:.2f}')

# Аналитическое решение
theta_normal = normalEqn(np.hstack((np.ones((m, 1)), X)), y)

# Нормировка для аналитического предсказания
normalized_engine_speed = (2104 - mu[0]) / sigma[0]  # Нормировка
normalized_num_gears = (3 - mu[1]) / sigma[1]  # Нормировка

predicted_price_normal = np.array([[1, normalized_engine_speed, normalized_num_gears]]) @ theta_normal
print(f'Предсказанная стоимость трактора (аналитическое решение): {predicted_price_normal[0]:.2f}')
