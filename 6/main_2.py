import numpy as np
import matplotlib.pyplot as plt

def decision_boundary_plot():
    # Параметры модели:
    # theta0=1, theta1=0, theta2=2, theta3=1, theta4=1, theta5=0
    theta0, theta1, theta2, theta3, theta4, theta5 = 1, 0, 2, 1, 1, 0

    # Шаг 1. Создаём сетку (x1, x2), на которой будем вычислять z.
    x1_vals = np.linspace(-12, 12, 300)
    x2_vals = np.linspace(-12, 12, 300)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Шаг 2. Вычисляем z на всей сетке.
    # z = theta0 + theta1*x1 + theta2*x2 + theta3*(x1*x2) + theta4*x1^2 + theta5*x2^2
    Z = (theta0
         + theta1 * X1
         + theta2 * X2
         + theta3 * (X1 * X2)
         + theta4 * X1**2
         + theta5 * X2**2)

    # Шаг 3. Строим контур по уровню z=0 (это и есть разделяющая кривая).
    plt.figure(figsize=(8, 6))
    # levels=[0] означает: строим контур(ы) по уровню z = 0
    contour = plt.contour(X1, X2, Z, levels=[0], colors='red', linewidths=2)
    contour.collections[0].set_label("decision boundary (z=0)")

    # Шаг 4. Заштриховываем области:
    #   - Z < 0 (класс 0)
    #   - Z > 0 (класс 1)
    # Для удобства используем plt.contourf c двумя уровнями.
    # levels = [-∞, 0, +∞] => область ниже 0 и область выше 0
    plt.contourf(X1, X2, Z, levels=[Z.min(), 0, Z.max()],
                 colors=['lightblue', 'lightgreen'], alpha=0.5)

    # Легенду для классов можно дополнительно подписать вручную:
    class_0_patch = plt.Rectangle((0,0), 0, 0, facecolor='lightblue', alpha=0.5,
                                  label='Class 0 (z < 0)')
    class_1_patch = plt.Rectangle((0,0), 0, 0, facecolor='lightgreen', alpha=0.5,
                                  label='Class 1 (z > 0)')
    plt.gca().add_patch(class_0_patch)
    plt.gca().add_patch(class_1_patch)

    # Подписи осей и легенда
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Разделяющая кривая для логистической регрессии (при alpha=0.5)")
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.savefig('gr.png')


if __name__ == "__main__":
    decision_boundary_plot()
