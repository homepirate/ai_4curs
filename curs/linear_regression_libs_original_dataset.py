import pandas as pd
import numpy as np
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt


# Загрузка данных
data_path = 'csvdata.csv'
df = pd.read_csv(data_path)

# Просмотр данных
print(df.head())


# Удаление индекса, если он не нужен
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Проверим и обработаем пропуски, если есть
print(f"Пропуски в данных: \n{df.isna().sum()}")
df = df.dropna()  # В данном случае удаляем строки с пропусками (если они есть)

# Преобразование категориальных признаков (City и Location) в числовые
label_enc_city = LabelEncoder()
label_enc_loc = LabelEncoder()

df['City'] = label_enc_city.fit_transform(df['City'])
df['Location'] = label_enc_loc.fit_transform(df['Location'])

# Целевая переменная (Price) и входные данные (все остальные колонки, кроме Price)
X = df.drop(columns=['Price'])
y = df['Price']

# Нормализация числовых данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Перемешивание и разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

print("Данные успешно обработаны!")


# Построение модели
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Выходной слой для регрессии
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

# Regularization and Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Model training with Early Stopping
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100,
                    batch_size=32, verbose=1, callbacks=[early_stopping])


# Оценка на тестовых данных
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Средняя абсолютная ошибка на тестовых данных: {test_mae:.2f}")

# График ошибок на обучающей и валидационной выборках

plt.plot(history.history['loss'], label='Обучающая выборка')
plt.plot(history.history['val_loss'], label='Валидационная выборка')
plt.title('График обучения модели')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка (MSE)')
plt.legend()
plt.savefig('error.png')

# Пример использования модели для прогнозирования
sample_input = np.array([[2, 645, 67, 1]])
sample_input_scaled = scaler.transform(sample_input)  # Нормализация данных
predicted_price = model.predict(sample_input_scaled)

print(f"Прогнозируемая цена: {predicted_price[0][0]:.2f}")


