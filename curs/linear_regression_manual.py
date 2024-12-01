import numpy as np
import csv


# Функция для нормализации данных (min-max нормализация)
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Функция для денормализации данных
def denormalize(normalized_data, original_data):
    return (normalized_data * (np.max(original_data) - np.min(original_data))) + np.min(original_data)


# Функция для вычисления MSE (среднеквадратичная ошибка)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Функция для предсказания
def predict(X, weights):
    return np.dot(X, weights)


# Реализация градиентного спуска
def gradient_descent(X, y, weights, learning_rate, epochs):
    m = len(y)  # Количество примеров
    for epoch in range(epochs):
        y_pred = predict(X, weights)  # Предсказания на текущем шаге
        error = y_pred - y  # Ошибка
        gradients = (1 / m) * np.dot(X.T, error)  # Градиент
        weights -= learning_rate * gradients  # Обновляем веса

        # Печатаем MSE каждые 100 итераций
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE: {mean_squared_error(y, y_pred)}")

    return weights


# Функция для загрузки данных из CSV
def load_data(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]

    X = []
    y = []
    for row in data:
        X.append([
            float(row['City']),  # Город
            float(row['Area']),  # Площадь
            float(row['Location']),  # Локация
            float(row['No. of Bedrooms'])  # Количество спален
        ])
        y.append(float(row['Price']))  # Цена

    return np.array(X), np.array(y)


# Основная функция
def main():
    # Загружаем данные
    file_path = 'shdf.csv'  # Путь к вашему CSV-файлу
    X, y = load_data(file_path)

    # Нормализация данных
    y_original = y.copy()  # Сохраняем оригиналы для денормализации
    y = normalize(y)
    X = normalize(X)

    # Добавляем столбец единиц для учета свободного члена (bias)
    X = np.c_[np.ones(X.shape[0]), X]

    # Инициализация параметров
    weights = np.zeros(X.shape[1])  # Инициализируем веса нулями
    learning_rate = 0.01
    epochs = 13

    # Обучение модели
    weights = gradient_descent(X, y, weights, learning_rate, epochs)

    # Вывод обученных весов
    print(f"Обученные веса: {weights}")

    # Пример предсказания
    example = np.array([2, 645, 67, 1])  # Пример входных данных (без цены)

    # Нормализация примера (учитываем только признаки, без bias)
    example_normalized = normalize(example)

    # Добавляем bias (единицу) для примера
    example_with_bias = np.concatenate(([1], example_normalized))

    # Предсказание
    prediction_normalized = predict(example_with_bias, weights)

    # Денормализация предсказания
    prediction = denormalize(prediction_normalized, y_original)
    print(f"Предсказание для примера: {prediction}")


if __name__ == '__main__':
    main()
