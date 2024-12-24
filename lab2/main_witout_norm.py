import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def loadData(filename):
    # Загрузка данных из файла
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, :-1].values  # Признаки (все столбцы, кроме последнего)
    y = data.iloc[:, -1].values   # Целевая переменная (последний столбец)
    return X, y


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


# Загрузка данных
X, y = loadData('ex1data2.txt')
m = len(y)  # Количество примеров

# Добавление единичного столбца для терма сдвига (bias term) для градиентного спуска
X_with_ones = np.hstack((np.ones((m, 1)), X))

# Обучение модели с использованием градиентного спуска без нормализации
alpha = 0.05  # Скорость обучения (маленькая из-за отсутствия нормализации)
iterations = 100    # Количество итераций
theta = np.zeros(X_with_ones.shape[1])  # Инициализация параметров θ нулями
theta_gd_no_norm, J_history_no_norm = gradientDescentMulti(X_with_ones, y, theta, alpha, iterations)

# Сохранение графика функции стоимости для модели без нормализации
plt.figure()
plt.plot(range(1, iterations + 1), J_history_no_norm, '-b', linewidth=2)
plt.xlabel('Итерации')
plt.ylabel('Стоимость J')
plt.title('Сходимость градиентного спуска без нормализации')
# plt.savefig('gradient_descent_cost_no_norm.png')
plt.close()
print("Модель обучена с использованием градиентного спуска без нормализации.")

# Обучение модели с использованием градиентного спуска с нормализацией (для сравнения)
# Нормализация признаков
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma
X_norm = np.hstack((np.ones((m, 1)), X_norm))
theta = np.zeros(X_norm.shape[1])
alpha_norm = 0.000000000001
iterations_norm = 1000
theta_gd_norm, J_history_norm = gradientDescentMulti(X_norm, y, theta, alpha_norm, iterations_norm)

# Сохранение графика функции стоимости для модели с нормализацией
plt.figure()
plt.plot(range(1, iterations_norm + 1), J_history_norm, '-r', linewidth=2)
plt.xlabel('Итерации')
plt.ylabel('Стоимость J')
plt.title('Сходимость градиентного спуска с нормализацией')
# plt.savefig('gradient_descent_cost_norm.png')
plt.close()
print("Модель обучена с использованием градиентного спуска с нормализацией.")

# Обучение модели с использованием нормального уравнения
theta_normal = normalEqn(X_with_ones, y)
print("Модель обучена с использованием нормального уравнения.")

# Ввод данных пользователем для предсказаний
print("\n--- Предсказание стоимости трактора ---")
engine_speed_input = float(input("Введите скорость двигателя: "))
num_gears_input = float(input("Введите количество передач: "))

# Подготовка пользовательских данных для модели без нормализации
user_features_no_norm = np.array([[1, engine_speed_input, num_gears_input]])

# Предсказание с использованием градиентного спуска без нормализации
predicted_price_gd_no_norm = user_features_no_norm @ theta_gd_no_norm
print(f"Предсказанная стоимость (Градиентный спуск без нормализации): {predicted_price_gd_no_norm[0]:.2f}")

# Нормализация пользовательских данных для модели с нормализацией
engine_speed_norm = (engine_speed_input - mu[0]) / sigma[0]
num_gears_norm = (num_gears_input - mu[1]) / sigma[1]
user_features_norm = np.array([[1, engine_speed_norm, num_gears_norm]])

# Предсказание с использованием градиентного спуска с нормализацией
predicted_price_gd_norm = user_features_norm @ theta_gd_norm
print(f"Предсказанная стоимость (Градиентный спуск с нормализацией): {predicted_price_gd_norm[0]:.2f}")

# Предсказание с использованием нормального уравнения
predicted_price_normal = user_features_no_norm @ theta_normal
print(f"Предсказанная стоимость (Нормальное уравнение): {predicted_price_normal[0]:.2f}")

# 3D-график для сравнения реальных данных и предсказаний
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Реальные данные
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Реальные значения')

# Предсказание градиентным спуском с нормализацией
ax.scatter(engine_speed_input, num_gears_input, predicted_price_gd_norm[0],
           color='red', s=100, label='Предсказание')

# Предсказание нормальным уравнением
ax.scatter(engine_speed_input, num_gears_input, predicted_price_normal[0],
           color='green', s=100, label='Предсказание нормальное уравнение')

# Метки и легенда
ax.set_xlabel('Скорость двигателя')
ax.set_ylabel('Количество передач')
ax.set_zlabel('Стоимость')
ax.set_title('Сравнение реальных значений и предсказаний')
ax.legend()

# Сохранение графика
plt.savefig('models_comparison_without_norm.png')
plt.close()