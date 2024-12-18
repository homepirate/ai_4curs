import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_statistics(column):
    n = len(column)
    mean = sum(column) / n
    std = (sum((x - mean) ** 2 for x in column) / n) ** 0.5
    return mean, std


def standardize(column):
    mean, std = calculate_statistics(column)
    standardized = (column - mean) / std
    return standardized, mean, std


input_file = 'ex1data2.txt'
data = pd.read_csv(input_file, header=None, names=["Скорость оборота", "Число передач", "Цена"])

standardized_data = pd.DataFrame()
for column in ["Скорость оборота", "Число передач"]:
    standardized_data[column], mean, std = standardize(data[column])

standardized_data["Цена"] = data["Цена"]

new_file = 'new_file.csv'
standardized_data.to_csv(new_file, index=False)
print(f"Нормализованные данные сохранены в файл: {new_file}")

X = np.c_[np.ones((data.shape[0], 1)), standardized_data[["Скорость оборота", "Число передач"]].values]
y = data["Цена"].values.reshape(-1, 1)


def predict(theta, X):
    return X.dot(theta)

def compute_cost(X, y, theta):
    m = len(y)
    predictions = predict(theta, X)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = predict(theta, X)
        errors = predictions - y
        theta = theta - (alpha / m) * X.T.dot(errors)
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

def plot_learning_rate(X, y, theta, alphas, iterations):
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        theta_opt, cost_history = gradient_descent(X, y, theta, alpha, iterations)
        plt.plot(range(iterations), cost_history, label=f"Alpha = {alpha}")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.title("Подбор скорости обучения (learning rate)")
    plt.legend()
    plt.grid(True)
    plt.savefig("learningrate.png")

initial_theta = np.zeros((X.shape[1], 1))
iterations = 1500
alphas = [0.01, 0.03, 0.1, 0.3, 1.0]

plot_learning_rate(X, y, initial_theta, alphas, iterations)

alpha_best = 0.03
theta_best, cost_history_best = gradient_descent(X, y, initial_theta, alpha_best, iterations)

print(f"Оптимальные параметры (градиентный спуск): {theta_best}")
print(f"Конечная стоимость (градиентный спуск): {cost_history_best[-1]}")


def matrix_multiply(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            value = sum(A[i][k] * B[k][j] for k in range(len(A[0])))
            row.append(value)
        result.append(row)
    return result

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def inverse_matrix(matrix):
    n = len(matrix)
    augmented = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(matrix)]

    for i in range(n):
        if augmented[i][i] == 0:
            for j in range(i + 1, n):
                if augmented[j][i] != 0:
                    augmented[i], augmented[j] = augmented[j], augmented[i]
                    break
        divisor = augmented[i][i]
        for j in range(2 * n):
            augmented[i][j] /= divisor

        for j in range(i + 1, n):
            multiplier = augmented[j][i]
            for k in range(2 * n):
                augmented[j][k] -= multiplier * augmented[i][k]

    for i in range(n - 1, -1, -1):
        for j in range(i):
            multiplier = augmented[j][i]
            for k in range(2 * n):
                augmented[j][k] -= multiplier * augmented[i][k]

    inverse = [row[n:] for row in augmented]
    return inverse


def normal_equation(X, y):
    Xt = transpose(X)
    XtX = matrix_multiply(Xt, X)
    XtX_inv = inverse_matrix(XtX)
    XtY = matrix_multiply(Xt, y)
    theta = matrix_multiply(XtX_inv, XtY)

    return theta

theta_normal = normal_equation(X.tolist(), y.tolist())

print(f"Параметры (нормальное уравнение): {theta_normal}")
print(f"Конечная стоимость (нормальное уравнение): {compute_cost(X, y, theta_normal)}")

def compare_models(theta_best, theta_normal, X, y):
    cost_best = compute_cost(X, y, theta_best)
    cost_normal = compute_cost(X, y, theta_normal)

    print(f"Стоимость при оптимальных параметрах (градиентный спуск): {cost_best}")
    print(f"Стоимость при параметрах (нормальное уравнение): {cost_normal}")

    plt.figure(figsize=(10, 6))
    plt.plot(y, label="Истинные значения", color='blue')
    plt.plot(predict(theta_best, X), label="Предсказания (Градиентный спуск)", color='red', linestyle='--')
    plt.plot(predict(theta_normal, X), label="Предсказания (Нормальное уравнение)", color='green', linestyle='-.')
    plt.xlabel("Индекс наблюдения")
    plt.ylabel("Цена")
    plt.legend()
    plt.title("Сравнение результатов градиентного спуска и нормального уравнения")
    plt.grid(True)
    plt.savefig("alpha.png")

compare_models(theta_best, theta_normal, X, y)

def predict_new_data(theta, mean_std, X_old, y_old):
    print("\nВведите новые данные для предсказания:")
    speed = float(input("Скорость оборота: "))
    gears = float(input("Число передач: "))

    speed_std = (speed - mean_std['Скорость оборота'][0]) / mean_std['Скорость оборота'][1]
    gears_std = (gears - mean_std['Число передач'][0]) / mean_std['Число передач'][1]

    X_new = np.array([1, speed_std, gears_std]).reshape(1, -1)

    predicted_price = predict(theta, X_new)[0][0]
    print(f"Предсказанная цена: {predicted_price:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(y_old, label="Истинные значения", color='blue')
    plt.plot(predict(theta, X_old), label="Предсказания (Градиентный спуск)", color='red', linestyle='--')

    plt.scatter(len(y_old), predicted_price, color='purple', label="Новая точка (предсказание)", zorder=5)
    plt.xlabel("Индекс наблюдения")
    plt.ylabel("Цена")
    plt.legend()
    plt.title("Добавление нового предсказания на график")
    plt.grid(True)
    plt.savefig("predict.png")

mean_std = {
    'Скорость оборота': calculate_statistics(data['Скорость оборота']),
    'Число передач': calculate_statistics(data['Число передач']),
}

predict_new_data(theta_best, mean_std, X, y)