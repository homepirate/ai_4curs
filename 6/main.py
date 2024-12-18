import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# Заданные параметры
theta = np.array([1, 0, 2, 1, 1, 0])
alpha = 0.5

# Определяем диапазон для x1 и x2
x1_min, x1_max = -5, 5
x2_min, x2_max = -5, 5

# Создаем сетку значений
x1 = np.linspace(x1_min, x1_max, 400)
x2 = np.linspace(x2_min, x2_max, 400)
X1, X2 = np.meshgrid(x1, x2)

# Вычисляем значение функции на сетке
# Модель: theta0 + theta1*x1 + theta2*x2 + theta3*x1*x2 + theta4*x1^2 + theta5*x2^2
F = theta[0] + theta[1]*X1 + theta[2]*X2 + theta[3]*X1*X2 + theta[4]*X1**2 + theta[5]*X2**2

# Построение графика
plt.figure(figsize=(8, 6))

# Закрашивание областей
# Класс 1: F >= alpha
# Класс 0: F < alpha
plt.contourf(X1, X2, F >= alpha, alpha=0.3, colors=['#FFAAAA', '#AAFFAA'])

# Контур, где F = alpha
contour = plt.contour(X1, X2, F, levels=[alpha], colors='k')
plt.clabel(contour, fmt={alpha: 'Граница решения'}, inline=True)

# Добавление легенды
legend_elements = [
    Patch(facecolor='#AAFFAA', edgecolor='k', label='Класс 1'),
    Patch(facecolor='#FFAAAA', edgecolor='k', label='Класс 0'),
    Patch(facecolor='none', edgecolor='k', label='Граница решения')
]
plt.legend(handles=legend_elements, loc='upper left')

# Добавление меток осей и заголовка
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Граница решения и области классов')

plt.grid(True)
plt.savefig('1.png')
