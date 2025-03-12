import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Импортируем matplotlib для построения графиков
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Загрузка датасета
data_path = '../../data/csvdata.csv'
data = pd.read_csv(data_path)

# Определение признаков и целевой переменной
feature_columns = ['Area', 'No. of Bedrooms', 'City', 'Location']
target_column = 'Price'

# Разделение данных
X = data[feature_columns]
y = data[target_column]

# Перемешивание данных
def shuffle_dataset(X, y, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)

X, y = shuffle_dataset(X, y, random_state=42)

# Разделение на обучающую, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.3333, random_state=42)  # 0.3333 * 0.3 ≈ 0.1

# Определение признаков для числовых и категориальных данных
numerical_features = ['Area', 'No. of Bedrooms']
categorical_features = ['City', 'Location']

# Создание трансформеров
numerical_transformer = MinMaxScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Создание ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Создание Pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Обучение модели
pipeline.fit(X_train, y_train)

# Валидация модели
val_predictions = pipeline.predict(X_val)
val_rmse = mean_squared_error(y_val, val_predictions, squared=False)

# Тестирование модели
test_predictions = pipeline.predict(X_test)
test_rmse = mean_squared_error(y_test, test_predictions, squared=False)

# Сохранение Pipeline
joblib.dump(pipeline, 'random_forest_pipeline.pkl')

# Отображение результатов
results = {
    "Validation RMSE": val_rmse,
    "Test RMSE": test_rmse
}

print("Model Performance:", results)

# Добавление графика обучения (Learning Curve) с одной линией (валидационная RMSE)
train_sizes, train_scores, val_scores = learning_curve(
    estimator=pipeline,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Преобразование отрицательных MSE в RMSE для валидационных данных
val_rmse_curve = np.sqrt(-val_scores)

# Средние значения и стандартные отклонения для валидационной RMSE
val_rmse_mean = np.mean(val_rmse_curve, axis=1)
val_rmse_std = np.std(val_rmse_curve, axis=1)

# Определение границ для оси Y с небольшим отступом
y_min = max(val_rmse_mean - val_rmse_std) * 0.95  # Немного ниже минимального значения
y_max = max(val_rmse_mean + val_rmse_std) * 1.05  # Немного выше максимального значения

# Построение графика обучения с одной линией
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, val_rmse_mean, 'o-', color='green', label='Validation RMSE')
plt.fill_between(train_sizes, val_rmse_mean - val_rmse_std,
                 val_rmse_mean + val_rmse_std, alpha=0.1, color='green')
plt.title('Learning Curve for RandomForestRegressor')
plt.xlabel('Training Set Size')
plt.ylabel('RMSE')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()

# Настройка масштаба оси Y
plt.ylim(y_min, y_max)

plt.show()
