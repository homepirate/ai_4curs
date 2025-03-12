import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename):
    data = pd.read_csv(filename, header=None)
    X = data.iloc[:, :-1].values  # Признаки
    y = data.iloc[:, -1].values  # Целевая переменная
    return X, y

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def compute_cost(X, y, theta):
    """
    Вычисление функции стоимости для линейной регрессии.
    """
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    J = (1 / (2 * m)) * np.sum(errors ** 2)
    return J

def gradient_descent_with_early_stop(X, y, theta, alpha, max_iters, tolerance=1e-7):
    """
    Реализация градиентного спуска с ранней остановкой.
    Остановка происходит, если разница между предыдущей и текущей стоимостью меньше tolerance.
    """
    m = len(y)
    J_history = []
    for i in range(max_iters):
        # Вычисление градиентов
        gradients = (X.T @ (X @ theta - y)) / m
        theta -= alpha * gradients
        J_current = compute_cost(X, y, theta)
        J_history.append(J_current)

        # Условие остановки: если изменение функции стоимости меньше tolerance
        if i > 0 and abs(J_history[-2] - J_current) < tolerance:
            return theta, J_history, i + 1

    return theta, J_history, max_iters

def plot_cost_history(J_history, name):
    """
    Построение графика функции стоимости по итерациям.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(J_history) + 1), J_history, '-b', linewidth=2)
    plt.xlabel('Итерации')
    plt.ylabel('Функция стоимости J(Theta)')
    plt.title('Сходимость градиентного спуска')
    plt.grid(True)
    plt.savefig(f'cost_history_{name}.png')

def plot_iterations_vs_alpha(alpha_values, iterations):
    """
    Построение графика количества итераций для достижения точности 10^-7 в зависимости от alpha.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(alpha_values, iterations, '-o', linewidth=2, markersize=6)
    plt.xscale('log')  # Логарифмическая шкала по оси X для alpha
    plt.xlabel('Значение alpha (логарифмическая шкала)')
    plt.ylabel('Количество итераций')
    plt.title('Зависимость количества итераций от alpha')
    plt.grid(True)
    plt.savefig('iterations_vs_alpha.png')

# Основной блок
if __name__ == "__main__":
    # 1. Загрузка данных
    X, y = load_data('ex1data2.txt')
    m = len(y)

    # 2. Нормализация признаков
    X_norm, mu, sigma = feature_normalize(X)

    # 3. Добавление столбца единиц для интерсепта
    X_norm = np.hstack((np.ones((m, 1)), X_norm))

    # 4. Список значений alpha
    alpha_values = [0.001, 0.01, 0.05]
    max_iters = 10000  # Максимальное число итераций
    tolerance = 1e-7  # Критерий точности функции стоимости

    iterations_to_converge = []  # Хранение количества итераций для каждого alpha

    print("Запуск градиентного спуска для разных значений alpha...")
    for alpha in alpha_values:
        theta_initial = np.zeros(X_norm.shape[1])  # Начальные значения параметров
        theta_optimal, J_history, num_iters = gradient_descent_with_early_stop(X_norm, y, theta_initial, alpha, max_iters, tolerance)
        iterations_to_converge.append(num_iters)
        print(f"Alpha = {alpha}, Итерации для точности 10^-7: {num_iters}")

    # 5. Построение графика зависимости количества итераций от alpha
    plot_iterations_vs_alpha(alpha_values, iterations_to_converge)

    # 6. Пример работы с оптимальным alpha
    best_alpha = 0.05
    theta_initial = np.zeros(X_norm.shape[1])
    theta_optimal, J_history, _ = gradient_descent_with_early_stop(X_norm, y, theta_initial, best_alpha, max_iters, tolerance)

    # 7. Построение графика функции стоимости для лучшего alpha
    plot_cost_history(J_history, 1)

    best_alpha = 0.01
    theta_initial = np.zeros(X_norm.shape[1])
    theta_optimal, J_history, _ = gradient_descent_with_early_stop(X_norm, y, theta_initial, best_alpha, max_iters,
                                                                   tolerance)

    # 7. Построение графика функции стоимости для лучшего alpha
    plot_cost_history(J_history, 2)

    # 8. Вывод найденных параметров
    print("Оптимальные параметры (Theta):", theta_optimal)


# α = 0.001 α=0.001: Очень медленная сходимость, так как шаги слишком малы. Потребуется огромное количество итераций, чтобы минимизировать J(theta)
# α = 0.01 α=0.01: Сходимость лучше, чем у 0.001 0.001, но все еще медленнее, чем у 0.05 0.05. Выигрыш в стабильности, но снижение в эффективности.
# α = 0.1 α=0.1 и α = 0.3 α=0.3 Более быстрые темпы уменьшения J ( thena) J(theta) в начале, но со временем возникает нестабильность из-за слишком больших шагов.

