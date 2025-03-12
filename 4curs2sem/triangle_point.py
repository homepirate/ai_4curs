import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

def check_place_for_point(mask, coord, r):
    """Проверяет, можно ли разместить точку (окружность радиуса r) по координате coord (row, col)."""
    height, width = mask.shape
    Y, X = np.ogrid[:height, :width]
    circle_mask = (Y - coord[0])**2 + (X - coord[1])**2 <= r**2
    return not np.any(mask[circle_mask])

def draw_point(mask, coord, r):
    """Рисует точку (заполняет область круга) по координате coord (row, col)."""
    height, width = mask.shape
    Y, X = np.ogrid[:height, :width]
    circle_mask = (Y - coord[0])**2 + (X - coord[1])**2 <= r**2
    mask[circle_mask] = 1
    return mask

def get_polygon_mask(mask_shape, vertices):
    """
    Создаёт булеву маску для многоугольника.
    vertices — массив вершин в формате (row, col).
    Для Path нужно передать координаты как (x, y), поэтому меняем порядок.
    """
    height, width = mask_shape
    y_indices, x_indices = np.mgrid[0:height, 0:width]
    points = np.vstack((x_indices.ravel(), y_indices.ravel())).T
    # Меняем порядок координат: (row, col) -> (col, row)
    poly_path = Path(np.array([[v[1], v[0]] for v in vertices]))
    mask_poly = poly_path.contains_points(points).reshape(mask_shape)
    return mask_poly

def check_place_for_triangle(mask, vertices):
    """Проверяет, можно ли разместить треугольник с заданными вершинами (многоугольник) без пересечения с другими объектами."""
    tri_mask = get_polygon_mask(mask.shape, vertices)
    return not np.any(mask[tri_mask])

def draw_triangle(mask, vertices):
    """Рисует треугольник (заполняет область многоугольника) с заданными вершинами."""
    tri_mask = get_polygon_mask(mask.shape, vertices)
    mask[tri_mask] = 1
    return mask

# Задаём параметры холста и объектов
canvas_size = 600
mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

n_objects = 40       # всего объектов
placed_objects = 0   # размещённых объектов
max_attempts = 1000  # ограничение на число попыток, чтобы избежать бесконечного цикла
attempts = 0

r_point = 5          # радиус точки
side_triangle = 40   # сторона равностороннего треугольника
R_triangle = side_triangle / np.sqrt(3)  # расстояние от центра до вершины (описанная окружность)

while placed_objects < n_objects and attempts < max_attempts:
    obj_type = np.random.choice(['point', 'triangle'])
    if obj_type == 'point':
        # Для точки выбираем координаты так, чтобы круг полностью был внутри холста
        x = np.random.randint(r_point, canvas_size - r_point)
        y = np.random.randint(r_point, canvas_size - r_point)
        coord = (y, x)  # (row, col)
        if check_place_for_point(mask, coord, r_point):
            mask = draw_point(mask, coord, r_point)
            placed_objects += 1
    else:  # triangle
        # Для треугольника выбираем центр так, чтобы фигура не выходила за границы (отступ не меньше R_triangle)
        x = np.random.randint(int(R_triangle), canvas_size - int(R_triangle))
        y = np.random.randint(int(R_triangle), canvas_size - int(R_triangle))
        center = (y, x)  # (row, col)
        angle = np.random.uniform(0, 2*np.pi)
        # Вычисляем вершины равностороннего треугольника через центр и угол
        vertices = np.array([
            [center[0] + R_triangle * np.cos(angle + 2*np.pi*k/3),
             center[1] + R_triangle * np.sin(angle + 2*np.pi*k/3)]
            for k in range(3)
        ])
        # Если какая-либо вершина выходит за границы, пропускаем попытку
        if (vertices[:,0] < 0).any() or (vertices[:,0] >= canvas_size).any() or \
           (vertices[:,1] < 0).any() or (vertices[:,1] >= canvas_size).any():
            attempts += 1
            continue
        if check_place_for_triangle(mask, vertices):
            mask = draw_triangle(mask, vertices)
            placed_objects += 1
    attempts += 1

print(f"Размещено объектов: {placed_objects} (попыток: {attempts})")

plt.figure(figsize=(6,6))
plt.imshow(mask, cmap='gray', origin='upper')
plt.title("40 объектов: точки и треугольники")
plt.axis('off')
plt.show()
