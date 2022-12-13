import math
import numpy as np
import matplotlib.pyplot as plt


def A(a1, a2):
    """Створення матриці А з заданими параметрами а1 та а2"""
    A = np.zeros((3, 3))
    A[0][1] = 1
    A[1][2] = 1
    A[2][0] = -1
    A[2][1] = -a1
    A[2][2] = -a2
    return A


def B(b):
    """Створення матриці B з заданим параметром b"""
    B = np.zeros((3, 1))
    B[0][0] = 0
    B[1][0] = 0
    B[2][0] = b
    return B


def C():
    """Створення матриці C"""
    C = []
    for i in range(3):
        value = np.zeros((1, 3))
        if i != 0:
            value[0][i] = 0
        else:
            value[0][i] = 1
        C.append(value)
    return C


def F(x, t, q):
    """Обрахуємо F = e^(A*T0)"""
    I = np.eye(3)  # одинична матриця n*n
    for i in range(1, q + 1):
        I += np.linalg.matrix_power(x.dot(t), i) / math.factorial(i)
    return I


def G(x, q, t, b):
    """Обрахуємо G = I*B*(T0 + (A*T0^2)/2! + ... + (A^q-1*T0^q)/q!)"""
    b1 = B(b)
    I = np.eye(3)
    tmp = b1*I
    for j in range(2, q):
        tmp += (np.linalg.matrix_power(x, j - 1).dot(I) * (t ** j)) / math.factorial(j)
    return tmp.dot(b1)


def formulaX(f, x, u, g):
    """Обрахуємо Xk+1 = F*Xk + G*Uk"""
    res = f.dot(x) + g * u
    return res


def X():
    x = np.zeros((3, 1))
    x[0][0] = 0
    x[1][0] = 0
    x[2][0] = 0
    return x


def formulaY(x, c, y):
    """Обрахуємо y = C*x"""
    y.append(c.dot(x))
    return y


def variables():
    """Задаємо параметри"""
    a1 = float(input('Введіть a1 для матриці A: '))
    a2 = float(input('Введіть a2 для матриці A: '))
    b = float(input('Введіть b для матриці B: '))
    t0 = float(input('Введіть T0:'))
    q = int(input('Введіть q:'))
    var = [a1, a2, t0, q, b]

    return var


def main():
    vars = variables()
    vec = X()
    t = list(np.arange(0, 200 + vars[2], vars[2]))
    a = A(vars[0], vars[1])
    f = F(a, vars[2], vars[3])
    g = G(a, vars[3], vars[2], vars[4])
    x = []
    x.append(vec)
    option = int(input('Виберіть один варіант з трьох: '))
    if option == 1:
        u = 1
        for i in range(len(t)):
            vec = formulaX(f, vec, u, g)
            x.append(vec)
    elif option == 2:
        k0 = 15//vars[2]
        u = 1
        for i in range(len(t)):
            vec = formulaX(f, vec, u, g)
            x.append(vec)
            if t[i] == k0:
                u = -1
    elif option == 3:
        k0 = 15//vars[2]
        u = 1
        k = 0
        for i in range(len(t)):
            vec = formulaX(f, vec, u, g)
            x.append(vec)
            if k == k0:
                u = -1
            elif k == 2 * k0:
                u = -1
            elif k == 3 * k0:
                u = 1
            k += 1
    else:
        print("Немає такого варіанту")
        return 0

    y = []
    for i in x:
        y = formulaY(i, C()[0], y)
    list_x = []
    for i in y:
        a = i.tolist()
        list_x.append(a[0])
    list_y = []
    for i in range(len(list_x) - 1):
        list_y.append(list_x[i][0])


    """Графік"""
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.grid()
    plt.xticks(np.arange(t[0], t[-1] + 1, 20))
    plt.plot(t, list_y, color='red')
    plt.show()


main()


"""Параметри для дослідження"""
# 1) a1 = 1, a2 = 10, b = 1, t0 = 0.0001, q = 10
# 2) a1 = 1, a2 = 1, b = 1, t0 = 0.0001, q = 10
# 3) a1 = 10, a2 = 1, b = 1, t0 = 0.0001, q = 10

