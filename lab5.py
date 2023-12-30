import sympy as smp
import numpy as np
from prettytable import PrettyTable


def max_m(f: smp.core.add.Add, a, b):
    x = smp.Symbol('x')
    x_arr = np.linspace(a, b + 0.0000001, 1000)
    f2 = []
    for xi in x_arr:
        f2.append(np.abs(f.subs(x, xi).evalf()))
    return float(max(f2))


def trap(h):
    int_trap = 0
    y = []
    for xi in np.arange(a, b + 0.0000001, h):
        y.append(F.subs(x, xi).evalf())

    for i in range(1, len(y) - 1):
        int_trap += y[i]

    int_trap += (y[0] + y[len(y) - 1])/2
    int_trap *= h
    return int_trap


def simpson(h):
    int_simp = 0
    y = []
    for xi in np.arange(a, b + 0.0000001, h):
        y.append(F.subs(x, xi).evalf())

    for i in range(1, len(y) - 1, 2):
        int_simp += 4 * y[i]

    for i in range(2, len(y) - 2, 2):
        int_simp += 2 * y[i]

    int_simp += y[0] + y[len(y) - 1]
    int_simp *= h / 3
    return int_simp


if __name__ == "__main__":
    print("\033[3m\033[31m{}\033[0m".format("Определение шага интегрирования\n"))
    x = smp.Symbol("x")
    F = smp.cos(x) * smp.exp(-x)
    accuracy = 0.001
    a = 0
    b = 2
    f1 = smp.diff(F, x)
    f2 = smp.diff(f1, x)
    h = (b - a) / 4
    M = max_m(f2, a, b)
    while M * np.abs(b - a) * h ** 2 / 12 >= accuracy:
        h /= 4
    print("\033[3m\033[32m{}\033[0m".format(f'Шаг интегрирования h ={h}\n'))

    print("\033[3m\033[31m{}\033[0m".format("Вычисление интеграла по формуле трапеций\n"))
    In = trap(h)
    print("\033[3m\033[34m{}\033[0m".format(f'С шагом h = {In}\n'))
    I2n = trap(2 * h)
    print("\033[3m\033[34m{}\033[0m".format(f'С шагом 2h = {I2n}\n'))
    print("\033[3m\033[34m{}\033[0m".format(f'Оценка погрешности = {(1 / 3) * np.abs(In - I2n)}\n'))
    print("\033[3m\033[34m{}\033[0m".format(f'Вычисленный интеграл: { smp.integrate(F, (x, a, b))}'
                                            f' = {smp.integrate(F, (x, a, b)).evalf()}\n'))

    print("\033[3m\033[31m{}\033[0m".format("Вычисление интеграла по формуле Симпсона\n"))
    In = simpson(h)
    print("\033[3m\033[34m{}\033[0m".format(f'С шагом h = {In}\n'))
    I2n = simpson(2 * h)
    print("\033[3m\033[34m{}\033[0m".format(f'С шагом 2h = {I2n}\n'))
    print("\033[3m\033[34m{}\033[0m".format(f'Оценка погрешности = {(1/15)*np.abs(In - I2n)}\n'))
    print("\033[3m\033[34m{}\033[0m".format(f'Вычисленный интеграл: {smp.integrate(F, (x, a, b))}'
                                            f' = {smp.integrate(F, (x, a, b)).evalf()}\n'))

    print("\033[3m\033[31m{}\033[0m".format("Вычисление интеграла по формуле Ньютона-Лейбница\n"))
    F_int = smp.integrate(F, x)
    print("\033[3m\033[34m{}\033[0m".format(f'Неопределенный интеграл: {F_int}\n'))
    print("\033[3m\033[34m{}\033[0m".format(f'{F_int.subs(x, b)} - {F_int.subs(x, a)} = {F_int.subs(x, b) - F_int.subs(x, a)} ={(F_int.subs(x, b) - F_int.subs(x, a)).evalf()}\n'))


