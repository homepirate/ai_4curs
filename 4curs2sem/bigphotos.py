import os
import cv2
import matplotlib.pyplot as plt


def load_images(directory, count=None):

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(directory) if f.lower().endswith(valid_extensions)]
    files.sort()  # сортируем для последовательности
    if count:
        files = files[:count]

    images = []
    for file in files:
        img_path = os.path.join(directory, file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
        else:
            print(f"Не удалось загрузить изображение: {img_path}")
    return images


def scale_image(image, scale_factor):

    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    scaled = cv2.resize(image, (new_width, new_height))
    return scaled


large_images_dir = "/large_images"  # директория с большими изображениями
modified_images_dir = "/modified_images"  # директория с модифицированными изображениями

large_images = load_images(large_images_dir, count=5)
modified_images = load_images(modified_images_dir)
modified_images = modified_images[:5]  # если нужно ровно 5

scale_factor = 0.5

large_scaled_sizes = []
modified_scaled_sizes = []

for img in large_images:
    scaled = scale_image(img, scale_factor)
    height, width = scaled.shape[:2]
    area = width * height  # площадь в пикселях
    large_scaled_sizes.append(area)

# Обработка модифицированных изображений
for img in modified_images:
    scaled = scale_image(img, scale_factor)
    height, width = scaled.shape[:2]
    area = width * height
    modified_scaled_sizes.append(area)

indices = list(range(1, len(large_scaled_sizes) + 1))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(indices, large_scaled_sizes, marker='o', label='Большие изображения')
plt.plot(indices, modified_scaled_sizes, marker='o', label='Модифицированные изображения')
plt.xlabel('Индекс изображения')
plt.ylabel('Размер (площадь в пикселях)')
plt.title('Зависимость индекса изображения от размера масштабированных изображений')
plt.legend()
plt.grid(True)
plt.show()
