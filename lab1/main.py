import numpy as np
import matplotlib.pyplot as plt


def warm_up_exercise(n):
    A = np.eye(n)

    A_manual = [[0 for _ in range(n)] for i in range(n)]
    for i in range(n):
        A_manual[i][i] = 1

    return A, A_manual

def plot_data(X, y):
    plt.scatter(X, y, marker='x', color='r')
    plt.xlabel('Number of Cars')
    plt.ylabel('Profit')
    plt.title('Profit vs. Number of Cars')
    plt.savefig('1.png')


def compute_cost(X, y, theta):
    # # Векторизированный расчет стоимости
    m = len(y)
    predict = np.dot(X, theta)
    errors = predict - y
    J = (1 / (2 * m)) * np.dot(errors, errors)
    return J


def compute_cost_by_element(X, Y, theta):
    m = len(Y)
    total_cost = 0

    for i in range(m):
        prediction = X[i].dot(theta)  # Предсказание для i-й строки
        error = prediction - Y[i]      # Ошибка для i-й строки
        total_cost += error ** 2      # Квадрат ошибки

    cost = (1 / (2 * m)) * total_cost
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        # Векторизированный шаг градиентного спуска
        theta = theta - (alpha / m) * (X.T.dot(X.dot(theta) - y))

    return theta


def gradient_descent_by_element(X, y, theta, alpha, iterations):
    m = len(y)
    n = len(theta)

    for _ in range(iterations):
        temp_theta = np.copy(theta)

        for j in range(n):
            sum_error = 0

            for i in range(m):
                hypothesis = np.dot(X[i], theta)
                error = hypothesis - y[i]
                sum_error += error * X[i][j]

            temp_theta[j] -= (alpha / m) * sum_error

        theta = temp_theta

    return theta


def work(theta):
    print("Программа прогнозирования прибыли запущена.")
    while True:
        user_input = input("Введите количество автомобилей (или 'exit' для выхода): ")
        if user_input.lower() == 'exit':
            break
        try:
            num_cars = float(user_input)
            profit = theta[0] + theta[1] * num_cars
            print(f'Предсказанная прибыль для {num_cars} автомобилей: {profit}')
        except ValueError:
            print("Пожалуйста, введите корректное значение.")


def main():

    print(warm_up_exercise(int(input("Введите размер матрицы: "))))
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    X = data[:, 0]  # Количество автомобилей
    y = data[:, 1]  # Прибыль
    m = len(y)  # Число примеров

    plot_data(X, y)

    X = np.c_[np.ones(m), X]
    theta = np.zeros(2) # Инициализируем параметры theta
    theta = np.array([1, 2])

    # Вычисление начальной стоимости
    print(f'Initial cost: {compute_cost(X, y, theta)}')
    print(f'Initial cost by element: {compute_cost_by_element(X, y, theta)}')

    alpha = 0.01
    iterations = 2000

    theta = gradient_descent(X, y, theta, alpha, iterations)

    print(f'Theta found by gradient descent: {theta}')

    theta_el = gradient_descent_by_element(X, y, theta, alpha, iterations)
    print(f'Theta found by gradient descent by element: {theta_el}')

    work(theta)


if __name__ == "__main__":
    main()
