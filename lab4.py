import matplotlib.pyplot as plt
import numpy as np
import sympy as sm
import sympy.core.add
from sympy import Symbol, sympify
import pandas as pd
from prettytable import PrettyTable
from scipy.optimize import fsolve


def eqs(vars):
    x, y = vars
    eq1 = 1 - np.sin(x)/2 - y
    eq2 = (0.7 - np.cos(y-1))/2 - x
    return [eq1, eq2]


def equasion1(x):
    return 1 - np.sin(x)/2


def equasion2(y):
    return (0.7 - np.cos(y-1))/2


def div_method(inter, eq: sympy.core.add.Add, accuracy):
    """
    Метод половинного деления
    :param inter: интервал в виде массива [a, b]
    :param eq: уравнение
    :param accuracy: точность
    :return: таблица и xn
    """
    table = PrettyTable(["i", "a", "b", "xn", "|xn - xn-1|"])
    a = inter[0]
    b = inter[1]
    counter = 0
    xn = 0
    prev = 0
    flag = True
    while flag:
        xn = (a + b)/2
        if counter >=1:
            table.add_row([counter, a, b, xn, abs(xn - prev)])
        else:
            table.add_row([counter, a, b, xn, "-"])
        if (eq.subs(x, xn) < 0 and eq.subs(x, a) > 0) or (eq.subs(x, xn) > 0 and eq.subs(x, a) < 0):
            b = xn
        elif (eq.subs(x, xn) < 0 and eq.subs(x, b) > 0) or (eq.subs(x, xn) > 0 and eq.subs(x, b) < 0):
            a = xn
        if counter >= 1:
            if abs(xn - prev) < accuracy:
                flag = False
        counter += 1
        prev = xn
    return [table, xn]


def chord_method(inter, eq: sympy.core.add.Add, accuracy):
    """
        Метод половинного деления
        :param inter: интервал в виде массива [a, b]
        :param eq: уравнение
        :param accuracy: точность
        :return: таблица и xn
    """
    a = inter[0]
    b = inter[1]
    table = PrettyTable(["i", "a", "b", "xn", "|xn - xn-1|"])
    counter = 0
    xn = 0
    prev = 0
    flag = True
    while flag:
        h = -1 * eq.subs(x, a) * (b-a)/(eq.subs(x, b) - eq.subs(x, a))
        xn = a + h
        if (eq.subs(x, a) > 0 and eq.subs(x, xn) < 0) or (eq.subs(x, a) < 0 and eq.subs(x, xn) > 0):
            b = xn
        elif (eq.subs(x, b) > 0 and eq.subs(x, xn) < 0) or (eq.subs(x, b) < 0 and eq.subs(x, xn) > 0):
            a = xn
        if counter >= 1:
            table.add_row([counter, a, b, xn, abs(xn - prev)])
        else:
            table.add_row([counter, a, b, xn, "-"])
        if counter >= 1:
            if abs(xn - prev) < accuracy:
                flag = False
        prev = xn
        counter += 1
    return [table, xn]


def last_method(inter, eq: sympy.core.add.Add, accuracy, xn):
    """
            Метод последовательных приближений
            :param inter: интервал в виде массива [a, b]
            :param eq: уравнение
            :param accuracy: точность
            :return: таблица и xn
    """
    a = inter[0]
    b = inter[1]
    table = PrettyTable(["i", "xn", "|xn - xn-1|"])
    counter = 0
    table.add_row([counter, xn, "-"])
    prev = xn
    flag = True
    counter += 1
    while flag:
        xn = -eq.subs(x, prev)
        table.add_row([counter, xn, abs(xn - prev)])
        if abs(xn - prev) < accuracy:
            flag = False
        prev = xn
        counter += 1
    return [table, xn]


def newton_method(inter, eq: sympy.core.add.Add, accuracy):
    """
            Метод Ньютона
            :param inter: интервал в виде массива [a, b]
            :param eq: уравнение
            :param accuracy: точность
            :return: таблица и xn
    """
    a = inter[0]
    b = inter[1]
    table = PrettyTable(["i", "xn", "|xn - xn-1|"])
    counter = 0
    xn = a
    flag = True
    flag2 = True
    y_der1 = sm.diff(eq, x)
    y_der2 = sm.diff(y_der1, x)
    step = 0.02
    while a <= xn <= b and flag2:
        if eq.subs(x, xn) * y_der2.subs(x, xn) > 0:
            flag2 = False
        else:
            xn += step
    table.add_row([counter, xn, "-"])
    prev = xn
    counter += 1
    while flag:
        xn = prev - eq.subs(x, prev) / y_der1.subs(x, prev)
        table.add_row([counter, xn, abs(xn - prev)])
        if abs(xn - prev) < accuracy:
            flag = False
        prev = xn
        counter += 1
    return [table, xn]


if __name__ == "__main__":
    #Задание1
    print("\033[3m\033[31m{}\033[0m".format("Задание1\n"))
    x = np.linspace(-5, 3)
    plt. plot(x, -1.38*x**3 - 5.42 * x**2 + 2.57 * x + 10.95)
    plt.grid(True)
    plt.show()
    inter1 = [-4, -3]
    inter2 = [-2, 0]
    inter3 = [1, 2]
    x = Symbol("x")
    y = -1.38 * x**3 - 5.42 * x**2 + 2.57 * x + 10.95
    accuracy = 0.001

    solution1 = div_method(inter1, y, accuracy)
    print("\033[3m\033[34m{}\033[0m".format("Метод половинного деления\n"))
    print(f'Интервал 1 [{inter1[0], inter1[1]}] \n', solution1[0], "\n",
          f'f({solution1[1]}) = {y.subs(x, solution1[1])}')
    solution2 = div_method(inter2, y, accuracy)
    print(f'Интервал 2 [{inter2[0], inter2[1]}] \n', solution2[0], "\n",
          f'f({solution2[1]}) = {y.subs(x, solution2[1])}')
    solution3 = div_method(inter3, y, accuracy)
    print(f'Интервал 3 [{inter3[0], inter3[1]}] \n', solution3[0], "\n",
          f'f({solution3[1]}) = {y.subs(x, solution3[1])}')
    print("\033[3m\033[32m{}\033[0m".format(f'Проверка: {sm.solve(y)}\n'))

    print("\033[3m\033[34m{}\033[0m".format("Метод хорд\n"))
    solution1 = chord_method(inter1, y, accuracy)
    print(f'Интервал 1 [{inter1[0], inter1[1]}] \n', solution1[0], "\n",
          f'f({solution1[1]}) = {y.subs(x, solution1[1])}')
    solution2 = chord_method(inter2, y, accuracy)
    print(f'Интервал 2 [{inter2[0], inter2[1]}] \n', solution2[0], "\n",
          f'f({solution2[1]}) = {y.subs(x, solution2[1])}')
    solution3 = chord_method(inter3, y, accuracy)
    print(f'Интервал 3 [{inter3[0], inter3[1]}] \n', solution3[0], "\n",
          f'f({solution3[1]}) = {y.subs(x, solution3[1])}')
    print("\033[3m\033[32m{}\033[0m".format(f'Проверка: {sm.solve(y)}\n'))

    print("\033[3m\033[34m{}\033[0m".format("Метод Ньютона\n"))
    solution1 = newton_method(inter1, y, accuracy)
    print(f'Интервал 1 [{inter1[0], inter1[1]}] \n', solution1[0], "\n",
          f'f({solution1[1]}) = {y.subs(x, solution1[1])}')
    solution2 = newton_method(inter2, y, accuracy)
    print(f'Интервал 2 [{inter2[0], inter2[1]}] \n', solution2[0], "\n",
          f'f({solution2[1]}) = {y.subs(x, solution2[1])}')
    solution3 = newton_method(inter3, y, accuracy)
    print(f'Интервал 3 [{inter3[0], inter3[1]}] \n', solution3[0], "\n",
          f'f({solution3[1]}) = {y.subs(x, solution3[1])}')
    print("\033[3m\033[32m{}\033[0m".format(f'Проверка: {sm.solve(y)}\n'))

    print("\033[3m\033[34m{}\033[0m".format("Метод последовательных приближений\n"))
    y2 = ((- 5.42 * x**2 + 2.57 * x + 10.95)/-1.38)**(1/3)
    solution1 = last_method(inter1, y2, accuracy, -3.7)
    print(f'Интервал 1 [{inter1[0], inter1[1]}] \n', solution1[0], "\n",
          f'f({solution1[1]}) = {y.subs(x, solution1[1])}')

    #Задание 2
    print("\033[3m\033[31m{}\033[0m".format("Задание2\n"))
    x_vals = np.linspace(-4, 5, 400)
    y_vals = np.linspace(-4, 5, 400)

    plt.plot(x_vals, equasion1(x_vals), label='sin (x+1)−y=1.2')
    plt.plot(equasion2(y_vals), y_vals, label='2x+cos(y)=2')

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True)

    plt.title('Графики функций')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    solution = fsolve(eqs, (0, 0))
    print("Решение системы уравнений:", solution)

    x, y = sympy.symbols('x y')
    F = sympy.sin(x) - 2 - 2*y
    G = sympy.cos(y-1) - 0.7 + 2 * x

    f_dx = sympy.diff(F, x)
    f_dy = sympy.diff(F, y)
    g_dx = sympy.diff(G, x)
    g_dy = sympy.diff(G, y)

    J = sympy.Matrix([[f_dx, f_dy], [g_dx, g_dy]])

    print("Матрица Якоби:\n", J)

    dx, dy = sympy.symbols('dx dy')

    linear_system = sympy.Eq(J * sympy.Matrix([dx, dy]), -sympy.Matrix([F, G]))

    print("Система линейных уравнений:\n")
    print(linear_system)

    x_0 = 100
    y_0 = 100

    x_n = 0
    y_n = 4

    k = 0
    # пока не достигнем нужной точности
    while (sympy.sqrt((x_0 - x_n) ** 2 + (y_0 - y_n) ** 2) >= 0.001):
        k += 1

        # Подстановка значений x и y
        J_subs = J.subs([(x, x_n), (y, y_n)])
        rhs_vector = sympy.Matrix([F, G])
        rhs_vector_subs = rhs_vector.subs([(x, x_n), (y, y_n)])

        # Решение системы
        solution = J_subs.inv() * (-rhs_vector_subs)

        print("\nИтерация - ", k)
        print("\ndelta_x =", solution[0].evalf())
        print("delta_y =", solution[1].evalf())

        x_0 = x_n
        y_0 = y_n
        x_n = x_n + solution[0].evalf()
        y_n = y_n + solution[1].evalf()

        print("x =", x_n)
        print("y =", y_n)

    print("\nРешение:")
    print("x =", x_n)
    print("y =", y_n)





