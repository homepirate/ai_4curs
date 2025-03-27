import cv2

def draw_aruco(image, corners, ids):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            pts = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = pts
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # Вычисляем центр квадрата
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            # Определяем параметры текста
            text = str(markerID)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            # Сдвигаем текст так, чтобы он был центрирован по центру квадрата
            text_offset_x = cX - text_width // 2
            text_offset_y = cY + text_height // 2

            cv2.putText(image, text, (text_offset_x, text_offset_y), font, font_scale, (0, 255, 0), thickness)
        cv2_imshow(image)
        cv2.waitKey(0)

img = cv2.imread("aruco.jpg")
draw_aruco(img.copy(), corners, ids)

# Запрос пути к изображению (формат JPG)

# Преобразуем изображение в оттенки серого для детекции
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Инициализируем словарь ArUco и параметры детектора
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters_create()

# Обнаруживаем маркеры
corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Отображаем изображение с размеченными маркерами
draw_aruco(img.copy(), corners, ids)
