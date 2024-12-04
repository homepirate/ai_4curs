import numpy as np
import matplotlib.pyplot as plt


def warm_up_exercise(n):
    A = np.eye(n)

    A_manual = [[0 for _ in range(n)] for i in range(n)]
    for i in range(n):
        A_manual[i][i] = 1

    return A, A_manual

def plot_data(X, y, theta=None, filename='1.png'):
    plt.scatter(X, y, marker='x', color='r')
    plt.xlabel('Number of Cars')
    plt.ylabel('Profit')
    plt.title('Profit vs. Number of Cars')

    if theta is not None:
        x_line = np.linspace(min(X), max(X), len(X))
        y_line = theta[0] + theta[1] * x_line

        plt.plot(x_line, y_line, color='b', label='Линия регрессии')
        plt.legend()
    plt.savefig(filename)


def compute_cost(X, y, theta):
    # # Векторизированный расчет стоимости
    m = len(y)
    predict = np.dot(X, theta)
    errors = predict - y
    J = (1 / (2 * m)) * np.dot(errors, errors)
    return J


def compute_cost_by_element(X, Y, theta):
    m = len(Y)  # Число примеров
    total_cost = 0

    for i in range(m):

        error = 0
        for j in range(len(theta)):  # Для каждого параметра
            error += ((theta[j] * X[i][j]) - Y[i] / len(theta))

        # error -= Y[i]

        # Суммируем квадрат ошибки
        total_cost += error ** 2

    # Возвращаем стоимость (среднеквадратичная ошибка)
    cost = (1 / (2 * m)) * total_cost
    return cost


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        # Векторизированный шаг градиентного спуска
        theta = theta - (alpha / m) * (X.T.dot(X.dot(theta) - y))

    return theta


def gradient_descent_by_element(X, y, theta, alpha, iterations):
    m = len(y)  # Число примеров
    n = len(theta)  # Число параметров

    for _ in range(iterations):
        temp_theta = np.copy(theta)

        # Для каждого параметра theta
        for j in range(n):
            sum_error = 0

            # Для каждого примера
            for i in range(m):

                error = 0
                for k in range(0, n):
                    error += (theta[k] * X[i][k] - y[i] / n)  # Умножаем соответствующие элементы высчитываем ошибку

                # Ошибка для i-го примера
                # error -= y[i]

                # Суммируем ошибки для каждого параметра
                sum_error += error * X[i][j - 1] if j > 0 else error

            # Обновляем параметр theta[j]
            temp_theta[j] -= (alpha / m) * sum_error

        theta = temp_theta

    return theta


def work(theta, theta_el):
    print("Программа прогнозирования прибыли запущена.")
    while True:
        user_input = input("Введите количество автомобилей (или 'exit' для выхода): ")
        if user_input.lower() == 'exit':
            break
        try:
            num_cars = float(user_input)
            profit = theta[0] + theta[1] * num_cars
            print(f'Предсказанная прибыль для {num_cars} автомобилей (векторный способ): {profit}')

            profit = theta_el[0] + theta[1] * num_cars
            print(f'Предсказанная прибыль для {num_cars} автомобилей (поэлементный способ): {profit}')

        except ValueError:
            print("Пожалуйста, введите корректное значение.")


def main():

    print(warm_up_exercise(int(input("Введите размер матрицы: "))))
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    X = data[:, 0]  # Количество автомобилей
    y = data[:, 1]  # Прибыль
    m = len(y)  # Число примеров

    plot_data(X, y)

    X_original = X.copy()

    X = np.c_[np.ones(m), X]
    theta = np.zeros(2) # Инициализируем параметры theta

    # Вычисление начальной стоимости
    print(f'Initial cost: {compute_cost(X, y, theta)}')
    print(f'Initial cost by element: {compute_cost_by_element(X, y, theta)}')

    alpha = 0.01
    iterations = 2000

    theta = gradient_descent(X, y, theta, alpha, iterations)

    print(f'Theta gradient descent: {theta}')
    plot_data(X_original, y, theta, '2.png')


    theta_el = gradient_descent_by_element(X, y, theta, alpha, iterations)
    print(f'Theta gradient descent by element: {theta_el}')

    work(theta, theta_el)


if __name__ == "__main__":
    main()
