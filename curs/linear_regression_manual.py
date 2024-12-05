import pandas as pd
import numpy as np

# 1. Загрузка данных
df = pd.read_csv('shdf.csv')

# 2. Удаление выбросов (на основе IQR)
q1 = np.percentile(df['Price'], 25)
q3 = np.percentile(df['Price'], 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]

# 3. Преобразование признаков (логарифм площади)
df['Log_Area'] = np.log1p(df['Area'])

# 4. Кодирование категориальных признаков вручную (создание дамми-переменных для 'Location')
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# 5. Разделение данных на X (признаки) и y (целевую переменную)
X = df.drop(['Price', 'Area'], axis=1).values
y = df['Price'].values

# 6. Масштабирование признаков вручную (стандартизация)
X = np.array(X, dtype=float)  # Принудительное приведение всех значений к float

# Удаление строк с NaN или Inf значениями
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    print("Найдены NaN/Inf значения в X. Удаляем их.")
    X = X[~np.isnan(X).any(axis=1)]  # Удаляем строки с NaN
    X = X[~np.isinf(X).any(axis=1)]  # Удаляем строки с Inf

# Проверка размерности X
if X.ndim == 1:
    X = X.reshape(-1, 1)  # Преобразуем в двумерный массив, если X одномерный
elif X.ndim > 2:
    raise ValueError("X должно быть не более чем двумерным массивом")

# Масштабируем нормальные признаки
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_std = np.where(X_std == 0, 1e-10, X_std)  # Избегаем деления на 0
X_scaled = (X - X_mean) / X_std

# 7. Разделение на обучающие и тестовые данные
split_index = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 8. Линейная регрессия с регуляризацией (Ridge regression)
lambda_reg = 1e4  # Параметр регуляризации
I = np.eye(X_train.shape[1])  # Единичная матрица для регуляризации
I[0, 0] = 0  # Не регуляризируем intercept (свободный член)

# Решение через регуляризованный метод наименьших квадратов (Ridge regression)
X_transpose = X_train.T
beta = np.linalg.inv(X_transpose @ X_train + lambda_reg * I) @ (X_transpose @ y_train)

# 9. Предсказания
y_pred = X_test @ beta

# 10. Оценка модели
mse = np.mean((y_test - y_pred) ** 2)
print(f'Mean Squared Error: {mse}')

ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
print(f'R2 Score: {r2}')

# 11. Проверка предсказаний
print("\nПроверка предсказаний:")
test_results = pd.DataFrame({
    'Real Price': y_test,
    'Predicted Price': y_pred
})
print(test_results.head())  # Печать первых 5 строк
