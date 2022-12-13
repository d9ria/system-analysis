import numpy as np

option = 1


def raus_method(coefficients):
    "Критерій Рауса"
    n = len(coefficients) - 1
    m = None
    if n % 2 == 0:
        m = np.zeros((n, int((n + 2) / 2)))
    if n % 2 != 0:
        m = np.zeros((n + 1, int((n + 1) / 2)))
    for j in range(0, m.shape[1], 1):
        if 2 * j < len(coefficients):
            a1 = coefficients[j * 2]
        else:
            a1 = 0.0
        m[0][j] = a1
        if 2 * j + 1 < len(coefficients):
            a2 = coefficients[j * 2 + 1]
        else:
            a2 = 0.0
        m[1][j] = a2
    for i in range(2, m.shape[0], 1):
        f = m[i - 2][0] / m[i - 1][0]
        for j in range(0, m.shape[1], 1):
            flag = j + 1 < m.shape[1]
            if flag:
                b1 = m[i - 2][j + 1]
                b2 = m[i - 1][j + 1]
            else:
                b1 = 0.0
                b2 - 0.0
            r = b1 - f * b2
            m[i][j] = r

    return m


def find_coeffs(m):
    """Пошук коефіцієнтів характеристичного рівняння."""
    list_of_coefficients = [1.0]
    copy_of = m
    for i in range(0, m.shape[1], 1):
        a = -1.0 * np.trace(copy_of)/(i + 1)
        list_of_coefficients.append(a)
        I = np.eye(m.shape[1])
        b = copy_of + I * a
        copy_of = np.dot(m, b)
    return list_of_coefficients


def matrix(option):
    """Задані матриці."""
    m = dict()
    m[1] = np.array([
        [-3, 2.2, 0.8, 0],
        [0.4, -4.7, 6.8, 0.1],
        [0.3, -3.1, -4.4, 0.6],
        [-0.5, 0.8, 1.9, -1.3]
    ])
    m[2] = np.array([
        [7, 2.19, 0.8, 0],
        [0.4, 5.29, 6.79, 0.1],
        [0.29, -3.1, 5.59, 5.9],
        [-0.49, 0.8, 1.89, 8.69]
    ])

    m[3] = np.array([
        [5.012, 2.457, 2.507, 0.069],
        [0.691, 0.797, 7.463, 0.548],
        [0.223, -3.264, 1.166, 0.827],
        [-0.696, 0.421, 3.221, 7.691]
    ])

    m[4] = np.array([
        [0.7446, 0.1372, 0.0981, 0.0035],
        [0.0319, 0.5701, 0.4213, 0.0274],
        [0.0153, -0.1879, 0.5901, 0.0413],
        [-0.0376, 0.0348, 0.1533, 0.8846]
    ])
    m[5] = np.array([
        [-0.9, 3.1, -0.2],
        [-0.4, -2.5, 3.2],
        [1.1, -1.5, -3.1]
    ])
    m[6] = np.array([
        [-4, 0.5, 0.7, -1.2],
        [1, -6, 1, 1],
        [0.5, 1, -3.8, 3],
        [1.5, 0, 1.5, -4.5]
    ])
    m[7] = np.array([
        [-2, 0.1, 0.3, -0.4, 0],
        [0.3, -2.9, -0.5, 0.4, 0.1],
        [0.2, -0.5, -1.8, 1.5, 0.2],
         [0, 1.0, 0, -2.0, 0.6],
         [-1, 0, 0.1, 0, -1.3]
    ])
    m[8] = np.array([
        [-5.5, -2.0, 0.2, -31],
        [0.3, -3.1, 2.1, -0.1],
        [0.5, 3.4, -4.0, 0],
        [0.1, -0.3, 1.4, -2.1],
    ])
    m[9] = np.array([
        [7, 2.19, 0.8, 0, 0.967],
        [0.4, 5.29, 6.79, 0.1, -0.568],
        [0.29, -3.1, 5.59, 5.9, 1.0],
        [-0.49, 0.8, 1.89, 8.69, 0.87],
        [0.0153, -0.1879, 0.5901, 0.0413, 1.457],
    ])
    m[10] = np.array([
        [-5.5, -2.0, 0.2, -3.1, 8],
        [1.0, -6.0, 1.0, 1.0, 2.0],
        [0.5, 3.4, -4.0, 0.0, 7.0],
        [0.1, -0.3, 1.4, -2.1, 0.0],
        [0.0, 1.0, -2.0, 0.6, 3.0],
    ])
    return m.get(option)


def main(option):
    """Головна функція, що друкує отримані результати."""
    for i in range(10):
        print(option)
        m = matrix(option)
        print(m, "\n")
        coefficients = find_coeffs(m)
        print("Коефіцієнти характеристичного рівняння: ")
        print(coefficients, "\n")
        method = raus_method(coefficients)
        print("Матриця Рауса:")
        print(method, "\n")
        option += 1


main(option)
