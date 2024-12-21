import numpy as np
import matplotlib.pyplot as plt

# Шаг 1: Подготовка данных
# Оцифрованные координаты точек (пример, заменить реальными данными)
data = np.array([
    [2, 3, 1],  # Точка класса 1
    [1, 1, 1],  # Точка класса 1
    [4, 1, 1],  # Точка класса 1
    [6, 4, 0],  # Точка класса 0
    [7, 2, 0],  # Точка класса 0
    [5, 6, 0],  # Точка класса 0
])

# Пересчет экранных координат в реальные
# (если известны координаты осей и точки преобразования, заменить)

X = data[:, :2]  # Координаты точек (x1, x2)
y = data[:, 2]  # Классы точек (1 или 0)

# Добавляем единичный столбец для x0
X = np.hstack((np.ones((X.shape[0], 1)), X))


# Шаг 2: Реализация обучения персептрона
def train_perceptron(X, y, epochs=100):
    weights = np.random.uniform(-1, 1, X.shape[1])  # Случайные веса
    print(f"Начальные веса: {weights}")
    for epoch in range(epochs):
        error_count = 0
        for i in range(len(X)):
            z = np.dot(weights, X[i])  # Сумма
            output = 1 if z >= 0 else 0  # Ступенчатая активация
            error = y[i] - output
            if error != 0:
                error_count += 1
                weights += error * X[i]  # Правило Хебба
        if error_count == 0:  # Если ошибок нет, обучение завершено
            print(f"Алгоритм сошелся на эпохе: {epoch + 1}")
            break
    print(f"Конечные веса: {weights}")
    return weights


# Обучение
weights = train_perceptron(X, y)


# Шаг 3: Построение разделяющей прямой
def plot_decision_boundary(weights, X, y):
    plt.figure(figsize=(8, 6))
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i, 1], X[i, 2], color='blue', label='Class 1' if i == 0 else "")
        else:
            plt.scatter(X[i, 1], X[i, 2], color='red', label='Class 0' if i == 0 else "")

    # Построение разделяющей прямой
    x_min, x_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
    y_min = -(weights[0] + weights[1] * x_min) / weights[2]
    y_max = -(weights[0] + weights[1] * x_max) / weights[2]
    plt.plot([x_min, x_max], [y_min, y_max], 'k--', label='Decision Boundary')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Perceptron Decision Boundary')
    plt.grid()
    plt.savefig('1.png')


# Визуализация
plot_decision_boundary(weights, X, y)
