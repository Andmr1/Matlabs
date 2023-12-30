import numpy as np
from prettytable import PrettyTable


def scalar_matrix_mul(matrix_left, matrix_right):
    if matrix_left.shape[1] != matrix_right.shape[0]:
        print("Invalid shapes\n")
        return None
    result = np.zeros((matrix_left.shape[0], matrix_right.shape[1]), dtype=int)
    for i in range(matrix_left.shape[0]):
        for j in range(matrix_right.shape[1]):
            for k in range(matrix_right.shape[0]):
                result[i, j] += matrix_left[i, k] * matrix_right[k, j]
    return result


def vector_matrix_mul(matrix_left, matrix_right):
    if matrix_left.shape[1] != matrix_right.shape[0]:
        print("Invalid shapes\n")
        return None
    result = np.zeros((matrix_left.shape[0], matrix_right.shape[1]), dtype=int)
    for i in range(matrix_left.shape[0]):
        for j in range(matrix_right.shape[1]):
            result[i][j] = np.dot(matrix_left[i, :], matrix_right[:, j])
    return result


def matrix_norm_3(matrix):
    tmp_array = matrix[:, 0]
    result = 0
    for i in tmp_array:
        result += abs(i)
    for i in range(1, matrix.shape[1]):
        tmp_array = matrix[:, i]
        tmp = 0
        for j in tmp_array:
            tmp += abs(j)
        if result < tmp:
            result = tmp
    return result


def vector_norma2(vector):
    result = 0
    for i in range(vector.size):
        result += vector[i] ** 2
    result = np.sqrt(result)
    return result


def house_qr(matrix):
    (r, c) = np.shape(matrix)
    Q = np.identity(r)
    R = np.copy(matrix)
    for i in range(r - 1):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_1 = np.identity(r)
        Q_1[i:, i:] -= 2.0 * np.outer(v, v)
        R = np.dot(Q_1, R)
        Q = np.dot(Q, Q_1)
        return Q, R



def gui_qr(matrix):
    (r, c) = np.shape(matrix)
    Q = np.identity(r)
    R = np.copy(matrix)
    (rows, cols) = np.tril_indices(r, -1, c)
    for (row, col) in zip(rows, cols):
        if R[row, col] != 0:
            r_ = np.hypot(R[col, col], R[row, col])
            c = R[col, col] / r_
            s = -R[row, col] / r_
            G = np.identity(r)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s
            R = np.dot(G, R)
            Q = np.dot(Q, G.T)
    return Q, R


def solve(matrix, vector):
    x = np.zeros((vector.size, 1))
    x[0] = vector[0] / matrix[0, 0]
    for i in range(1, vector.size):
        x[i] = vector[i]
        tmp = 0
        for j in range(i):
            x[i] = x[i] - x[j] * matrix[i, j]
            tmp = j
        x[i] /= matrix[i, tmp + 1]
    return x


def solve_rev(matrix, vector):
    x = np.zeros((vector.size, 1))
    a, b = matrix.shape
    x[x.size - 1] = vector[vector.size - 1] / matrix[a - 1, b - 1]
    for i in range(vector.size - 2, -1, -1):
        x[i] = vector[i]
        tmp = 0
        for j in range(vector.size - 1, i, -1):
            x[i] = x[i] - x[j] * matrix[i, j]
            tmp = j
        x[i] /= matrix[i, tmp - 1]
    return x


def gramm_qr(matrix):
    Q = np.zeros_like(matrix)
    cnt = 0
    for a in matrix.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])
        e = u / np.linalg.norm(u)
        Q[:, cnt] = e
        cnt += 1
    R = np.dot(Q.T, matrix)
    return Q, R


def iteration_solve(matrix, vector, accuracy):
    table = PrettyTable(["k", "x_1", "x_2", "x_3", "norma(x_k - x_k-1)"])
    x_prev = np.array([])
    x_cur = np.array([])
    for i in range(vector.size):
        x_cur = np.append(x_cur, matrix[i][i] / vector[i])
        x_prev = np.append(x_prev, 0)
    table.add_row([0, x_cur[0], x_cur[1], x_cur[2], 0])
    stop_point = 1
    a,b = matrix.shape
    counter = 1
    while(stop_point > accuracy):
        for i in range(x_cur.size):
            x_prev[i] = x_cur[i]
        for i in range(vector.size):
            x_cur[i] = vector[i]
            tmp = 0
            for j in range(b):
                if i != j:
                    x_cur[i] -= matrix[i, j] * x_prev[j]
                else:
                    tmp = matrix[i, j]
            x_cur[i] /= tmp
        stop_point = np.linalg.norm((x_cur - x_prev), ord=np.inf)
        table.add_row([counter, x_cur[0], x_cur[1], x_cur[2], stop_point])
        counter += 1
    return table


if __name__ == "__main__":
    #задание 1
    print("Задание 1 \n")
    matrix = np.random.uniform(2, 4, (10, 10))
    print(matrix, "\n")
    string4 = matrix[3, :]
    column5 = matrix[:, 4]
    print(string4, "\n")
    print(column5, "\n")
    result = 0
    for i in range(string4.size):
        result += string4[i] * column5[i]
    print(result, "\n")
    #задание 2
    print("Задание 2 \n")
    matrix = np.random.randint(2, 7, (3, 3))
    matrix2 = np.random.randint(2, 7, (3, 3))
    matrix3 = scalar_matrix_mul(matrix, matrix2)
    print("matrix1 = \n", matrix, "\n")
    print("matrix2 = \n", matrix2, "\n")
    print("scalar mul = \n", matrix3, "\n")
    matrix3 = vector_matrix_mul(matrix, matrix2)
    print("vector mul = \n", matrix3, "\n")
    matrix3 = np.dot(matrix, matrix2)
    print("np.dot mul = \n", matrix3, "\n")
    #задание 3
    print("Задание 3 \n")
    vector = np.random.randint(-10, 10, 10)
    print(vector)
    print(vector_norma2(vector), "\n")
    print(np.linalg.norm(vector), "\n")
    #Задание 4
    print("Задание 4 \n")
    matrix = np.random.randint(-2, 7, (4, 4), dtype=int)
    print(matrix, "\n")
    print(matrix_norm_3(matrix), "\n")
    print(np.linalg.norm(matrix, ord=1), "\n")
    #задание 5
    print("Задание 5 \n")
    array = np.array(vector, dtype=float)
    y = np.array((array[4:]), dtype=float)
    print("y: ", y)
    if array[4] < 0:
        beta = np.linalg.norm(y, 2)
    else:
        beta = -1 * np.linalg.norm(y, 2)
    print("B:\n", beta)

    u = np.array((y[0] - beta, y[1], y[2], y[3], y[4], y[5]))
    print("u:\n", u)

    v = u / np.linalg.norm(u, 2)
    print("v:\n", v)

    E = np.eye(6, 6)
    v_transpose = v.reshape(-1, 1)

    H_s = E - 2 * v * v_transpose
    print("H':\n", H_s)

    H = np.eye(10, 10)
    row_start, row_end = 4, 10
    col_start, col_end = 4, 10
    H[row_start:row_end, col_start:col_end] = H_s
    print("H:\n", H)

    array_transpose = array.reshape(-1, 1)

    result = np.dot(H, array)
    print("Исходный вектор: ", array)
    print("РЕЗУЛЬТАТ:\n", result)
    #Задание 6
    print("Задание 6 \n")
    matrix = np.array(matrix, dtype=float)
    print(matrix, "\n")
    U = np.zeros((matrix.shape[0], matrix.shape[0]), float)
    L = np.identity(matrix.shape[0], float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i <= j:
                U[i, j] = matrix[i, j] - np.dot(L[i, :i], U[:i, j])
            if i > j:
                L[i, j] = (matrix[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]
    print(L, "\n")
    print(U, "\n")
    print(np.dot(L, U), "\n")
    #Задание 7
    print("Задание 7 \n")
    matrix = np.random.randint(2, 4, (4, 4))
    print(matrix, "\n")
    Q1, R1 = np.linalg.qr(matrix)
    print("Q = \n", Q1, "\n")
    print("R = \n", R1, "\n")
    print("Проверка1 \n", np.dot(Q1, R1), "\n")
    Q2, R2 = house_qr(matrix)
    print("Q = \n", Q2, "\n")
    print("R = \n", R2, "\n")
    print("Проверка 2 \n", np.dot(Q2, R2))
    Q3, R3 = gui_qr(matrix)
    print("Q = \n", Q3, "\n")
    print("R = \n", R3, "\n")
    print("Проверка 3 \n", np.dot(Q3, R3))
    # Часть 2
    print("Часть 2 \n", "Задание 1\n")
    matrix = np.tril(np.random.randint(0, 10, size=(5, 5)))
    vector = np.random.randint(-5, 5, 5)
    x = np.zeros((vector.size, 1))
    x[0] = vector[0] / matrix[0, 0]
    for i in range(1, vector.size):
        x[i] = vector[i]
        tmp = 0
        for j in range(i):
            x[i] = x[i] - x[j] * matrix[i, j]
            tmp = j
        x[i] /= matrix[i, tmp + 1]
    print(matrix, "\n")
    print(vector, "\n")
    print(x, "\n")
    print(solve(matrix, vector))
    print(np.linalg.solve(matrix, vector), "\n")
    print(np.dot(matrix, x))
    # Задание 2
    print("Задание 2\n")
    matrix = np.array([[8.2, -3.2, 14.2, 14.8], [5.6, -12.0, 15.0, -6.4], [5.7, 3.6, -12.4, -2.3], [6.8, 13.2, -6.3, -8.7]])
    vector = np.array([-8.4, 4.5, 3.3, 14.3])
    U = np.zeros((matrix.shape[0], matrix.shape[0]), float)
    L = np.identity(matrix.shape[0], float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            if i <= j:
                U[i, j] = matrix[i, j] - np.dot(L[i, :i], U[:i, j])
            if i > j:
                L[i, j] = (matrix[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]
    y = solve(L, vector)
    x = solve_rev(U, y)
    print(x)
    print(np.linalg.solve(matrix, vector), "\n")
    print(np.dot(matrix, x))
    #Задание 3
    print("Задание 3\n")
    Q, R = gramm_qr(matrix)
    print("Q = \n", Q, "\n")
    print("R = \n", R, "\n")
    B = np.dot(Q.T, vector)
    x = solve_rev(R, B)
    print(x, "\n")
    print(np.linalg.solve(matrix, vector), "\n")
    print(np.dot(matrix, x))
    #Задание 4
    print("Задание 4 \n")
    matrix = np.array([[2.7, 3.3, 1.3],
              [3.5, -1.7, 2.8],
              [4.1, 5.8, -1.7]])
    vector = np.array([2.1, 1.7, 0.8])
    vec = matrix[0]
    vec_sol = vector[0]
    matrix[0] = matrix[0] + matrix[1]
    matrix[1] = matrix[1] - matrix[2]
    matrix[2] = matrix[2] - vec - vec
    vector[0] += vector[1]
    vector[1] -= vector[2]
    vector[2] -= 2 * vec_sol
    print("Преобразованная матрица: \n", matrix, "\n")
    print("преобразованный вектор \n", vector, "\n")
    accuracy = 0.001
    print(np.linalg.solve(matrix, vector))
    print(iteration_solve(matrix, vector, accuracy), "\n")
    #Задание 5
    print("Задание 5 \n")
    matrix = np.array([[3.1, 2.8, 1.9],
                       [1.9, 3.1, 2.1],
                       [7.5, 3.8, 2.1],
                       [3.01, 0.33, 0.11]])
    vector = np.array([0.2, 2.1, 5.6, 0.13])
    print(np.dot(np.dot(np.linalg.inv(np.dot(matrix.T, matrix)), matrix.T), vector))


