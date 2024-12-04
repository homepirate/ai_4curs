import numpy as np
import matplotlib.pyplot as plt

# Задаем параметры модели
theta = np.array([1, 0, 2, 1, 1, 0])  # (theta0, theta1, theta2, theta3, theta4, theta5)

# Определим функцию, которая возвращает значение модели
def model(x1, x2):
    return (theta[0] +
            theta[1]*x1 +
            theta[2]*x2 +
            theta[3]*x1*x2 +
            theta[4]*x1**2 +
            theta[5]*x2**2)

# Создаем сетку значений для x1 и x2
x1_vals = np.linspace(-3, 3, 400)
x2_vals = np.linspace(-3, 3, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Вычисляем значения модели
Z = model(X1, X2)

# Создаем график разделяющей кривой
plt.figure(figsize=(10, 8))
contour = plt.contour(X1, X2, Z, levels=[0.5], colors='red')
plt.clabel(contour, inline=True, fontsize=10)

# Указываем области классов
plt.fill_between(x1_vals, -np.sqrt(3 - x1_vals**2), -3, color='blue', alpha=0.3, label='Класс 0 (y=0)')
plt.fill_between(x1_vals, np.sqrt(3 - x1_vals**2), 3, color='green', alpha=0.3, label='Класс 1 (y=1)')

# Параметры графика
plt.title('Кривая разделения классов')
plt.xlabel('x1')
plt.ylabel('x2')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')

# Легенда
plt.legend()
plt.grid()
plt.xlim(-3, 3)
plt.ylim(-3, 3)

# Показываем график
plt.savefig('graf.png')
