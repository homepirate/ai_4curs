import cv2
import numpy as np
import matplotlib.pyplot as plt

# Если у вас есть своё изображение, укажите путь, иначе создадим синтетическое изображение.
img_path = "aruco.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    # Создадим синтетическое изображение размером 600x600 с несколькими фигурами
    img = np.zeros((600, 600), dtype=np.uint8)
    cv2.circle(img, (300, 300), 100, 255, -1)
    cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
    cv2.line(img, (400, 100), (500, 200), 255, 5)

# Массив пороговых значений от 10 до 500
thresholds = np.arange(10, 501)
num_contours = []

# Для каждого порогового значения выполняем бинаризацию и поиск контуров
for thresh in thresholds:
    ret, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours.append(len(contours))

# Строим график зависимости числа контуров от порогового значения
plt.figure(figsize=(10, 6))
plt.plot(thresholds, num_contours, marker='o')
plt.xlabel("Пороговое значение")
plt.ylabel("Количество контуров")
plt.grid(True)
plt.show()
