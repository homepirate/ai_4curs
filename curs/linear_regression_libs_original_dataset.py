import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Функция для загрузки данных
def load_data(file_path):
    # Считываем данные из CSV файла
    data = pd.read_csv(file_path)
    return data

# Функция для подготовки данных
def prepare_data(data):
    # Выделяем признаки (X) и целевую переменную (y)
    X = data[['City', 'Area', 'Location', 'No. of Bedrooms']]
    y = data['Price'].values

    # Преобразуем категориальные признаки 'City' и 'Location' с помощью OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Используем одну категорию для избежания коллинеарности
    X_encoded = encoder.fit_transform(X[['City', 'Location']])

    # Объединяем закодированные данные с остальными признаками
    X = np.hstack((X_encoded, X[['Area', 'No. of Bedrooms']].values))

    # Нормализуем числовые признаки
    scaler = StandardScaler()
    X[:, -2:] = scaler.fit_transform(X[:, -2:])  # Нормализуем Area и No. of Bedrooms

    return X, y, encoder, scaler

# Основная функция
def main():
    # Путь к вашему CSV-файлу
    file_path = '/home/evgeniy/PycharmProjects/4curs_AI/data/csvdata.csv'  # обновите путь к вашему файлу, если он другой

    # Загружаем и подготавливаем данные
    data = load_data(file_path)
    X, y, encoder, scaler = prepare_data(data)

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создаем модель линейной регрессии
    model = LinearRegression()

    # Обучаем модель
    model.fit(X_train, y_train)

    # Предсказываем стоимость на тестовой выборке
    y_pred = model.predict(X_test)

    # Выводим среднеквадратичную ошибку
    mse = mean_squared_error(y_test, y_pred)
    print("Среднеквадратичная ошибка (MSE):", mse)

    # Пример предсказания для новых данных
    example = pd.DataFrame([["Bangalore", 3340, "JP Nagar Phase 1", 4]],
                            columns=['City', 'Area', 'Location', 'No. of Bedrooms'])
    example_encoded = encoder.transform(example[['City', 'Location']])

    # Подготовка входного примера
    example_data = np.hstack((example_encoded, example[['Area', 'No. of Bedrooms']].values))
    example_data[:, -2:] = scaler.transform(example_data[:, -2:])  # Нормализация

    # Предсказание
    predicted_price = model.predict(example_data)
    print("Предсказанная стоимость для нового объекта:", predicted_price[0])

if __name__ == '__main__':
    main()
