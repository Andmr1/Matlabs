import sympy as smp
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def FF(y, x):
    return x - y ** 2


if __name__ == "__main__":
    print("\033[3m\033[31m{}\033[0m".format("Выбор шага интегрирования по методу Ренге-Кутта\n"))
    epsilon = 0.0001
    x = smp.Symbol('x')
    y = smp.Symbol('y')
    F = x - y ** 2

    newTable = PrettyTable(["h", "np.abs(y1 - y1)/15"])

    a = 0
    b = 2
    h = 0.1

    y1 = F.subs([(x, a + h), (y, b)])
    y2 = F.subs([(x, a + 2 * h), (y, b)])
    newTable.add_row([h, np.abs(y1 - y2) / 15])
    while np.abs(y1 - y2) / 15 < epsilon:
        h *= 2
        y1 = F.subs(x, a + h)
        y2 = F.subs(x, a + 2 * h)
        newTable.add_row([h, np.abs(y1 - y2) / 15])

    print(newTable)

    n = (b - a) / h
    print(n)
    h2 = h * 2

    print("\033[3m\033[31m{}\033[0m".format("Решение задачи Коши методом Ренге-Кутта\n"))
    newTable = PrettyTable(["x", "y_h", 'y_2h', 'delta'])
    newTable.add_row([a, b, b, 0])
    xi = a
    y_h = 2
    y_2h = 2

    x_arr = [a]
    y_arr = [2]

    x_arr.append(a)
    y_arr.append(b)
    while xi <= b:
        xi += h
        x_arr.append(xi)
        k1 = h * F.subs([(x, xi), (y, y_h)])
        k2 = h * F.subs([(x, xi + h / 2), (y, y_h + k1 / 2)])
        k3 = h * F.subs([(x, xi + h / 2), (y, y_h + k2 / 2)])
        k4 = h * F.subs([(x, xi + h), (y, y_h + k3)])
        y_h += 1 / 6 * (k1 + k2 + k3 + k4)

        y_arr.append(y_h)
        if (xi % h2) == 0:
            k1 = h * F.subs([(x, xi), (y, y_2h)])
            k2 = h * F.subs([(x, xi + h / 2), (y, y_2h + k1 / 2)])
            k3 = h * F.subs([(x, xi + h / 2), (y, y_2h + k2 / 2)])
            k4 = h * F.subs([(x, xi + h), (y, y_2h + k3)])
            y_2h += 1 / 6 * (k1 + k2 + k3 + k4)

        newTable.add_row([xi, y_h, y_2h, np.abs(y_h - y_2h) / 15])

    print(newTable)

    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.scatter(x_arr, y_arr)
    plt.plot(x_arr, y_arr)
    plt.show()

    print("\033[3m\033[31m{}\033[0m".format("Решение задачи Коши иетодом Эйлера\n"))
    newTable = PrettyTable(["x", "y_h", 'y_2h', 'delta'])
    newTable.add_row([a, b, b, 0])
    xi = a
    y_h = 2
    y_2h = 2

    x_arr2 = [a]
    y_arr2 = [2]

    x_arr2.append(a)
    y_arr2.append(b)
    while xi <= b:
        xi += h
        x_arr2.append(xi)
        y_h += h * F.subs([(x, xi), (y, y_h)])
        y_2h += h * F.subs([(x, xi), (y, y_2h)])

        y_arr2.append(y_h)

        newTable.add_row([xi, y_h, y_2h, np.abs(y_h - y_2h) / 15])

    print(newTable)

    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    plt.scatter(x_arr, y_arr)
    plt.plot(x_arr, y_arr)

    plt.scatter(x_arr2, y_arr2)
    plt.plot(x_arr, y_arr2)
    plt.show()


    yy = odeint(FF, b, x_arr)
    newTable = PrettyTable(["x", "Ренге-Кутт", 'Эйлер', 'Питон'])
    for i in range(len(x_arr)):
        newTable.add_row([x_arr[i], y_arr[i], y_arr2[i], yy[i]])
    print(newTable)

    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    rg, = plt.plot(x_arr, y_arr, 'o-', label='Ренге-Кутт')
    eil, = plt.plot(x_arr, y_arr2, 's-', label='Эйлер')
    python, = plt.plot(x_arr, yy, '^-', label='Питон')

    # Используйте объекты линии в plt.legend
    plt.legend(handles=[rg, eil, python])
    plt.show()
