import sys
import joblib
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox,
    QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
)

# Пути к файлам (при необходимости обновите)
DATA_PATH = '../../data/csvdata.csv'  # Путь к вашему датасету
PIPELINE_PATH = 'random_forest_pipeline.pkl'

# Загрузка Pipeline
try:
    pipeline = joblib.load(PIPELINE_PATH)
except FileNotFoundError:
    print(f"Не удалось найти Pipeline по пути: {PIPELINE_PATH}")
    sys.exit(1)

# Загрузка данных для получения городов и районов
try:
    data = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Не удалось найти датасет по пути: {DATA_PATH}")
    sys.exit(1)

# Проверка наличия необходимых столбцов
required_columns = {'City', 'Location', 'Price', 'Area', 'No. of Bedrooms'}
if not required_columns.issubset(data.columns):
    missing = required_columns - set(data.columns)
    print(f"В датасете отсутствуют необходимые столбцы: {missing}")
    sys.exit(1)

# Извлечение уникальных городов и районов
cities = sorted(data['City'].dropna().unique())
cities_districts = {
    city: sorted(data[data['City'] == city]['Location'].dropna().unique())
    for city in cities
}

class RealEstatePredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Предсказание цены недвижимости")
        self.setGeometry(100, 100, 400, 300)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Поле для ввода площади
        area_layout = QHBoxLayout()
        area_label = QLabel("Площадь (кв.м):")
        self.area_input = QLineEdit()
        self.area_input.setPlaceholderText("Введите площадь")
        area_layout.addWidget(area_label)
        area_layout.addWidget(self.area_input)
        layout.addLayout(area_layout)

        # Поле для ввода количества комнат
        rooms_layout = QHBoxLayout()
        rooms_label = QLabel("Количество комнат:")
        self.rooms_input = QLineEdit()
        self.rooms_input.setPlaceholderText("Введите количество комнат")
        rooms_layout.addWidget(rooms_label)
        rooms_layout.addWidget(self.rooms_input)
        layout.addLayout(rooms_layout)

        # Выпадающий список для выбора города
        city_layout = QHBoxLayout()
        city_label = QLabel("Город:")
        self.city_combo = QComboBox()
        self.city_combo.addItems(cities)
        self.city_combo.currentTextChanged.connect(self.update_districts)
        city_layout.addWidget(city_label)
        city_layout.addWidget(self.city_combo)
        layout.addLayout(city_layout)

        # Выпадающий список для выбора района
        district_layout = QHBoxLayout()
        district_label = QLabel("Район:")
        self.district_combo = QComboBox()
        self.update_districts(self.city_combo.currentText())
        district_layout.addWidget(district_label)
        district_layout.addWidget(self.district_combo)
        layout.addLayout(district_layout)

        # Кнопка для предсказания
        self.predict_button = QPushButton("Предсказать цену")
        self.predict_button.clicked.connect(self.predict_price)
        layout.addWidget(self.predict_button)

        # Поле для отображения предсказанной цены
        result_layout = QHBoxLayout()
        result_label = QLabel("Предсказанная цена:")
        self.result_display = QLineEdit()
        self.result_display.setReadOnly(True)
        result_layout.addWidget(result_label)
        result_layout.addWidget(self.result_display)
        layout.addLayout(result_layout)

        self.setLayout(layout)

    def update_districts(self, city):
        self.district_combo.clear()
        districts = cities_districts.get(city, [])
        self.district_combo.addItems(districts)

    def predict_price(self):
        try:
            # Получение и проверка ввода
            area_text = self.area_input.text()
            rooms_text = self.rooms_input.text()

            if not area_text or not rooms_text:
                raise ValueError("Поля площади и количества комнат не могут быть пустыми.")

            area = float(area_text)
            rooms = int(rooms_text)

            if area <= 0 or rooms <= 0:
                raise ValueError("Площадь и количество комнат должны быть положительными числами.")

            city = self.city_combo.currentText()
            district = self.district_combo.currentText()

            # Создание DataFrame для предсказания
            input_data = pd.DataFrame({
                'Area': [area],
                'No. of Bedrooms': [rooms],
                'City': [city],
                'Location': [district]
            })

            # Предсказание
            predicted_price = pipeline.predict(input_data)[0]

            # Отображение результата
            self.result_display.setText(f"{predicted_price:.2f}")
        except ValueError as ve:
            QMessageBox.warning(self, "Ошибка ввода", f"Пожалуйста, введите корректные значения.\n{ve}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Произошла ошибка: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealEstatePredictor()
    window.show()
    sys.exit(app.exec())
