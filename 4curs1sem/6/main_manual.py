import numpy as np
import matplotlib.pyplot as plt

def manual_decision_boundary_plot_no_bridge():
    """
    1) "Ручной" расчёт z = 1 + 2*x2 + x1*x2 + x1^2 на сетке -> раскраска классов.
    2) Явный вывод x2 через x1: x2 = -(x1^2 + 1)/(x1 + 2), но
       - строим отдельно для x1 < -2 и x1 > -2".
    """

    # -- 1) ПАРАМЕТРЫ --
    x_min, x_max = -12, 12   # диапазон осей
    n_points = 300          # сетка для раскраски

    # -- 2) Готовим сетку (x1, x2) для раскраски классов --
    x1_grid = np.linspace(x_min, x_max, n_points)
    x2_grid = np.linspace(x_min, x_max, n_points)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    # -- 3) Считаем z на этой сетке --
    Z = 1 + 2*X2 + X1*X2 + X1**2

    # Класс 0, если z < 0; класс 1, если z >= 0
    classes = np.where(Z >= 0, 1, 0)

    # -- 4) Рисуем раскраску "ручную" (pcolormesh)
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(['lightblue', 'lightgreen'])  # 0->blue, 1->green

    plt.figure(figsize=(9, 7))
    plt.pcolormesh(X1, X2, classes, cmap=cmap, shading='auto')
    plt.xlim([x_min, x_max])
    plt.ylim([x_min, x_max])

    # -- 5) Построим кривую z=0 в виде x2 = -(x1^2 + 1)/(x1+2),
    #       но раздельно для x1 < -2 и x1 > -2.

    def boundary_x2(x1):
        """Формула x2 = -(x1^2 + 1)/(x1 + 2)."""
        return -(x1**2 + 1)/(x1 + 2)

    # ЛЕВАЯ ВЕТВЬ: x1 < -2
    x1_vals_left = np.linspace(x_min, -2, 600, endpoint=False)  # не включая -2
    x2_vals_left = []
    x1_plot_left = []
    for x1 in x1_vals_left:
        denom = x1 + 2
        if abs(denom) < 1e-14:
            continue
        x2 = boundary_x2(x1)
        # Проверим, чтобы x2 попадал в диапазон [x_min, x_max], иначе уходит за границы
        if x_min <= x2 <= x_max:
            x1_plot_left.append(x1)
            x2_vals_left.append(x2)

    # ПРАВАЯ ВЕТВЬ: x1 > -2
    x1_vals_right = np.linspace(-2, x_max, 600, endpoint=False)  # начиная от -2
    x2_vals_right = []
    x1_plot_right = []
    for x1 in x1_vals_right:
        denom = x1 + 2
        if abs(denom) < 1e-14:
            continue
        x2 = boundary_x2(x1)
        if x_min <= x2 <= x_max:
            x1_plot_right.append(x1)
            x2_vals_right.append(x2)

    # Рисуем обе ветви одной линией
    plt.plot(x1_plot_left, x2_vals_left, 'r-', linewidth=2)
    plt.plot(x1_plot_right, x2_vals_right, 'r-', linewidth=2,
             label="Граница решения (z=0)")

    # -- 6) Оформляем, подписываем
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Разделяющая кривая без 'моста' между ветвями")

    # Легенда
    class_0_patch = plt.Rectangle((0,0), 0, 0, facecolor='lightblue', alpha=0.8,
                                  label='Class 0 (z < 0)')
    class_1_patch = plt.Rectangle((0,0), 0, 0, facecolor='lightgreen', alpha=0.8,
                                  label='Class 1 (z > 0)')
    plt.gca().add_patch(class_0_patch)
    plt.gca().add_patch(class_1_patch)
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("manual_boundary.png", dpi=150)


if __name__ == "__main__":
    manual_decision_boundary_plot_no_bridge()
