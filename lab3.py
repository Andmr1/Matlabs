import sympy as sm
from sympy import Symbol, simplify
from sympy import symbols, expand
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import ln

import math


if __name__ == "__main__":
    #Задание 1
    print("Задание 1\n")
    points = np.array([[0.02, 1.02316], [0.08, 1.09590], [0.12, 1.14725],
                       [0.17, 1.21483], [0.23, 1.30120], [0.30, 1.40976]])
    x = sm.Symbol('x')
    L = []
    a, b = points.shape
    for i in range(a):
        val = points[i, 1]
        deli = 1
        for j in range(a):
            if i != j:
                val *= (x - points[j, 0])
                deli *= (points[i, 0] - points[j, 0])
        L.append(val/deli)
    L = expand(sum(L))
    print("L = ", L)
    points_of_interest = np.array([0.102, 0.114, 0.125, 0.203, 0.154])
    value_of_interest = np.array([])
    for i in range(points_of_interest.size):
        value_of_interest = np.append(value_of_interest, L.subs(x, points_of_interest[i]))
    space = np.linspace(0, 0.6, 1000)
    y = [L.subs(x, i) for i in space]
    plt.plot(space, y, label="Многочлен Лагранжа", color="blue")
    plt.scatter([i[0] for i in points], [i[1] for i in points], label="Узлы", color="red")
    plt.scatter(points_of_interest, value_of_interest, label="Предсказанные значения", color="green")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    for i in range(points_of_interest.size):
        print(f'L({points_of_interest[i]}) = {value_of_interest[i]}')
    #Задание 2
    print("\n Задание 2 \n")
    points = np.array([[0.101, 1.26183], [0.106, 1.27644], [0.111, 1.29122], [0.116, 1.306617],
                       [0.121, 1.32130], [0.126, 1.33660], [0.131, 1.35207], [0.136, 1.136773],
                       [0.141, 1.38357], [0.146, 1.39959], [0.151, 1.41579]])
    points_of_interest = np.array([0.1026, 0.1440, 0.099, 0.153])
    h = 0.005
    deltas = np.zeros((11, 11), dtype=float)
    a, b = points.shape
    for i in range(a):
        deltas[i, 0] = points[i, 1]
    for i in range(a - 1):
        deltas[i, 1] = points[i + 1, 1] - points[i, 1]
    a, b = deltas.shape
    for i in range(2, b):
        for j in range(b - i):
            deltas[j, i] = deltas[j + 1, i - 1] - deltas[j, i - 1]
    table = pd.DataFrame(deltas)
    print("Таблица конечных разностей \n", table)
    #Первая интерполяционная формула Ньютона
    P_1 = points[0, 1]
    for i in range(1, a):
        tmp = 1
        for j in range(i):
            tmp *= (x - points[j, 0])
        P_1 += (deltas[0, i] / (math.factorial(i) * pow(h, j+1))) * tmp
    P_1.expand(P_1)
    print(P_1, "\n")
    #Вторая интерполяционная формула Ньютона
    P_2 = points[10, 1]
    for i in range(1, a):
        tmp = 1
        for j in range(i):
            tmp *= (x - points[a - j - 1, 0])
        P_2 += (deltas[a - i - 1, i] / (math.factorial(i) * pow(h, i))) * tmp
    print(P_2)
    value_of_interest = np.array([])
    #Расчет
    for i in range(points_of_interest.size):
        if points_of_interest[i] - 0.101 < 0.151 - points_of_interest[i]:
            value_of_interest = np.append(value_of_interest, P_1.subs(x, points_of_interest[i]))
        else:
            value_of_interest = np.append(value_of_interest, P_2.subs(x, points_of_interest[i]))
    for i in range(value_of_interest.size):
        print(f'y({points_of_interest[i]}) = {value_of_interest[i]}')
    #Задание 3
    x_points = np.linspace(2.0, 2.9, 10)
    y_points = np.array([])
    for i in range(x_points.size):
        y_points = np.append(y_points, math.log(x_points[i] + 1))
    print("x \n", x_points)
    print("y \n", y_points)
    L3 = 0
    n3 = len(x_points)
    for i in range(n3):
        term = 1
        for j in range(n3):
            if j != i:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        L3 += y_points[i] * term

    L3 = sm.expand(L3)
    print("L = ", L3)
    expect = L3.subs(x, 2.5555)
    real = math.log(2.5555 + 1)
    print(expect)
    print(real)
    print("Погрешность = ", abs(expect - real))


