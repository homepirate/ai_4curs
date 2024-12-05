import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Загрузка данных
df = pd.read_csv('shdf.csv')

# Удаление выбросов (на основе цены)
q1 = np.percentile(df['Price'], 25)
q3 = np.percentile(df['Price'], 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]

# Преобразование признаков (логарифм площади)
df['Log_Area'] = np.log1p(df['Area'])

# Кодирование категориальных признаков (если присутствуют)
df = pd.get_dummies(df, columns=['Location'], drop_first=True)

# Разделение на фичи и целевую переменную
X = df.drop(['Price', 'Area'], axis=1)  # Убрали `Area`, заменили на `Log_Area`
y = df['Price']

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Обучение линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')
print("Коэффициенты модели:", model.coef_)
print("Свободный член (intercept):", model.intercept_)


# --- Проверка предсказаний ---
print("\nПроверка предсказаний:")
test_results = pd.DataFrame({'Real Price': y_test, 'Predicted Price': y_pred})
print(test_results.head())  # Печать первых 5 реальных и предсказанных значений
