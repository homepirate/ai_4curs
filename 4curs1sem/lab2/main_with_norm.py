import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def loadData(filename):
    # Загрузка данных из файла
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, :-1].values  # Признаки (все столбцы, кроме последнего)
    y = data.iloc[:, -1].values  # Целевая переменная (последний столбец)
    return X, y


def featureNormalize(X):
    # Нормализация признаков (стандартизация)
    mu = np.mean(X, axis=0)  # Среднее значение для каждого признака
    sigma = np.std(X, axis=0)  # Стандартное отклонение для каждого признака
    X_norm = (X - mu) / sigma  # Нормализация: (X - mu) / sigma
    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    # Вычисление функции стоимости для многомерной линейной регрессии
    m = len(y)  # Количество примеров
    J = (1 / (2 * m)) * np.sum(np.square(X @ theta - y))  # Формула: J(θ) = (1/2m) * Σ((Xθ - y)^2)
    return J


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    # Градиентный спуск для многомерной линейной регрессии
    m = len(y)  # Количество примеров
    J_history = np.zeros(num_iters)  # История значений функции стоимости
    for i in range(num_iters):
        # Обновление параметров θ по формуле: θ = θ - (α / m) * X^T * (Xθ - y)
        theta = theta - (alpha / m) * (X.T @ (X @ theta - y))
        # Сохранение значения функции стоимости для текущей итерации
        J_history[i] = computeCostMulti(X, y, theta)
    return theta, J_history


def normalEqn(X, y):
    # Нормальное уравнение для решения задачи линейной регрессии
    # Формула: θ = (X^T * X)^(-1) * X^T * y
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y  # Pseudoinverse для вычисления θ
    return theta


# Загрузка и обработка данных
X, y = loadData('ex1data2.txt')
m = len(y)  # Количество примеров

# Нормализация признаков
X_norm, mu, sigma = featureNormalize(X)

# Добавление единичного столбца для терма сдвига (bias term) для градиентного спуска
X_norm = np.hstack((np.ones((m, 1)), X_norm))

# Обучение модели с использованием градиентного спуска
alpha = 0.05  # Скорость обучения
iterations = 100  # Количество итераций
theta = np.zeros(X_norm.shape[1])  # Инициализация параметров θ нулями
theta_gd, J_history = gradientDescentMulti(X_norm, y, theta, alpha, iterations)

# Сохранение графика функции стоимости
plt.figure()
plt.plot(range(1, iterations + 1), J_history, '-b', linewidth=2)
plt.xlabel('Итерации')
plt.ylabel('Стоимость J')
plt.title('Сходимость градиентного спуска')
# plt.savefig('../plots/gradient_descent_cost.png')
plt.close()
print("Модель обучена с использованием градиентного спуска.")

# Обучение модели с использованием нормального уравнения
X_with_ones = np.hstack((np.ones((m, 1)), X))  # Добавление единичного столбца
theta_normal = normalEqn(X_with_ones, y)
print("Модель обучена с использованием нормального уравнения.")

# Ввод данных пользователем для предсказаний
print("\n--- Предсказание стоимости трактора ---")
engine_speed_input = float(input("Введите скорость двигателя: "))
num_gears_input = float(input("Введите количество передач: "))

# Нормализация пользовательских данных для градиентного спуска
engine_speed_norm = (engine_speed_input - mu[0]) / sigma[0]
num_gears_norm = (num_gears_input - mu[1]) / sigma[1]
user_features_gd = np.array([[1, engine_speed_norm, num_gears_norm]])

# Предсказание с использованием градиентного спуска
predicted_price_gd = user_features_gd @ theta_gd
print(f"Предсказанная стоимость (Градиентный спуск): {predicted_price_gd[0]:.2f}")

# Предсказание с использованием нормального уравнения
user_features_normal = np.array([[1, engine_speed_input, num_gears_input]])
predicted_price_normal = user_features_normal @ theta_normal
print(f"Предсказанная стоимость (Нормальное уравнение): {predicted_price_normal[0]:.2f}")

# 3D-график для сравнения реальных данных и предсказаний
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Реальные данные
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Реальные значения')

# Предсказание градиентным спуском
ax.scatter(engine_speed_input, num_gears_input, predicted_price_gd[0],
           color='red', s=100, label='Предсказание (Градиентный спуск)')

# Предсказание нормальным уравнением
ax.scatter(engine_speed_input, num_gears_input, predicted_price_normal[0],
           color='green', s=100, label='Предсказание (Нормальное уравнение)')

# Метки и легенда
ax.set_xlabel('Скорость двигателя')
ax.set_ylabel('Количество передач')
ax.set_zlabel('Стоимость')
ax.set_title('Сравнение реальных значений и предсказаний')
ax.legend()

# Сохранение графика
plt.savefig('models_comparison.png')
plt.close()