import csv


# Функция для чтения данных из csv-файла и преобразования их в числовой формат
def read_csv(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Пропускаем заголовок
        for row in reader:
            if row:  # Пропускаем пустые строки
                # Преобразуем числовые данные в float, пропуская текстовые значения
                data.append([
                    float(row[2]),  # Price
                    float(row[3]),  # Area
                    len(row[4]) if row[4] else 0,  # Простая числовая замена для Location
                    float(row[5])  # No. of Bedrooms
                ])
    return data


# Функция нормализации данных (мин-макс нормализация)
def normalize_data(data):
    normalized_data = []
    min_max = []
    for i in range(len(data[0])):  # По каждому столбцу
        column = [row[i] for row in data]
        min_val = min(column)
        max_val = max(column)
        min_max.append((min_val, max_val))
        normalized_column = [(x - min_val) / (max_val - min_val) for x in column]
        normalized_data.append(normalized_column)
    # Транспонируем обратно в формат списка списков
    normalized_data = list(map(list, zip(*normalized_data)))
    return normalized_data, min_max


# Обратная нормализация для предсказания
def denormalize_price(normalized_price, min_price, max_price):
    return normalized_price * (max_price - min_price) + min_price


# Функция разбиения на признаки и целевую переменную
def split_features_labels(data):
    features = [row[1:] for row in data]  # Остальные столбцы кроме цены
    labels = [row[0] for row in data]  # Первый столбец (цена)
    return features, labels


# Линейная регрессия с градиентным спуском
class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0

    def train(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0] * n_features  # Инициализация весов
        self.bias = 0  # Инициализация смещения

        # Градиентный спуск
        for _ in range(self.iterations):
            y_predicted = [self.predict(x) for x in X]

            # Обновляем веса и смещение
            dW = [0] * n_features
            for j in range(n_features):
                dW[j] = (-2 / n_samples) * sum((y[i] - y_predicted[i]) * X[i][j] for i in range(n_samples))

            dB = (-2 / n_samples) * sum(y[i] - y_predicted[i] for i in range(n_samples))

            # Шаг градиентного спуска
            self.weights = [self.weights[j] - self.learning_rate * dW[j] for j in range(n_features)]
            self.bias -= self.learning_rate * dB

    def predict(self, x):
        return sum(self.weights[j] * x[j] for j in range(len(x))) + self.bias


# Главная функция
def main():
    # 1. Загрузка данных
    filename = 'csvdata.csv'
    data = read_csv(filename)

    # 2. Нормализация данных
    normalized_data, min_max = normalize_data(data)
    min_price, max_price = min_max[0]  # Нужны для обратной нормализации

    # 3. Разделение данных на признаки и целевую переменную
    features, labels = split_features_labels(normalized_data)

    # 4. Обучение модели
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.train(features, labels)

    # 5. Пример предсказания (с использованием обученной модели)
    test_sample = features[0]  # Берем первый образец для теста
    predicted_price_normalized = model.predict(test_sample)
    predicted_price = denormalize_price(predicted_price_normalized, min_price, max_price)

    print(f"Предсказанная цена: {predicted_price}")
    print(f"Фактическая цена: {denormalize_price(labels[0], min_price, max_price)}")


if __name__ == "__main__":
    main()
