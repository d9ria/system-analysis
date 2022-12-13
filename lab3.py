import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import linalg


def matrix_A(a1, a2):
    """Створення матриці А з заданими параметрами а1 та а2"""
    A = np.zeros((3, 3))
    A[0][1] = 1
    A[1][2] = 1
    A[2][0] = -1
    A[2][1] = -a1
    A[2][2] = -a2
    return A


def matrix_B(b):
    """Створення матриці B з заданим параметром b"""
    B = np.zeros((3, 1))
    B[0][0] = 0
    B[1][0] = 0
    B[2][0] = b
    return B


def matrix_C():
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


def variables():
    """Задаємо параметри"""
    x0_1 = int(input('Початкове x0: '))
    x0 = np.array([[x0_1], [0], [0]])
    x_aim_1 = int(input("Задане х*: "))
    x_aim = np.array([[x_aim_1], [0], [0]])
    a1 = float(input('Введіть a1 для матриці A: '))
    a2 = float(input('Введіть a2 для матриці A: '))
    b = float(input('Введіть b для матриці B: '))
    k0 = int(input('Введіть k0: '))
    k0 -= 2
    t0 = float(input('Введіть T0:'))
    q = int(input('Введіть q:'))
    var = [x0, x_aim, a1, a2, b, k0, t0, q]
    return var


def F(a1, a2, t0, q):
    """F = I + A*T0 + (A*T0)^2/2! + ..."""
    a = matrix_A(a1, a2)
    I = np.eye(3)
    for i in range(1, q + 1):
        I += np.linalg.matrix_power(a.dot(t0), i) / math.factorial(i)
    return I


def G(a1, a2, b, t0, q):
    """Обрахуємо G = I*B*(T0 + (A*T0^2)/2! + ... + (A^q-1*T0^q)/q!)"""
    b1 = matrix_B(b)
    I = np.eye(3)
    a = matrix_A(a1, a2)
    tmp = b1*I
    for j in range(2, q):
        tmp += (np.linalg.matrix_power(a, j - 1).dot(I) * (t0 ** j)) / math.factorial(j)
    return tmp.dot(b1)


def inverse_F(a1, a2, t0, q):
    """F = I - A*T0 + (A*T0)^2/2! + ..."""
    a = matrix_A(a1, a2)
    I = np.eye(3) - np.linalg.matrix_power(a.dot(t0), 1) / math.factorial(1)
    for i in range(2, q + 1):
        I += np.linalg.matrix_power(a.dot(t0), i) / math.factorial(i)
    return I


def power_F(k0, a1, a2, t0, q):
    """F^(k0-1) = П(k-1)"""
    f = F(a1, a2, t0, q)
    n = 0
    I = np.eye(3)
    while n < k0:
        I = f.dot(I)
        n += 1
    return I


def sum_of_G(a1, a2, b, t0, k0, q):
    sum = 0
    n = 0
    f = F(a1, a2, t0, q)
    g = G(a1, a2, b, t0, q)
    inv_F = inverse_F(a1, a2, t0, q)
    list_of_g_j = []
    while n <= k0+1:
        G_j = f.dot(g)
        list_of_g_j.append(G_j)
        sum += (G_j.dot(G_j.transpose()))
        f = inv_F.dot(f)
        n += 1

    return sum, list_of_g_j


def L(a1, a2, b, t0, k0, q):
    """Обрахуємо матрицю L та обернену до неї"""
    inv_F = inverse_F(a1, a2, t0, q)
    power_f = power_F(k0, a1, a2, t0, q)
    sum, list_of_g_j = sum_of_G(a1, a2, b, t0, k0, q)
    L = power_f.dot(sum)
    inv_L = linalg.inv(L)
    return L, inv_L


def L0(a1, a2, b, t0, k0, q, x_aim):
    """l0 = L^-1*x*"""
    l, inv_L = L(a1, a2, b, t0, k0, q)
    l0 = inv_L.dot(x_aim)
    return l0


def U_k(a1, a2, b, t0, k0, q, x_aim):
    """U_k = G.transpose(k)*l0"""
    u_k = []
    l0 = L0(a1, a2, b, t0, k0, q, x_aim)
    sum, list_of_g_j = sum_of_G(a1, a2, b, t0, k0, q)
    for k in range(0, k0+2):
        g_t = list_of_g_j[k].transpose()
        u_k.append(g_t.dot(l0))
    return u_k


def formula_X(x, f, g, u_k):
    res = f.dot(x) + g.dot(u_k)
    return res


def formula_Y(x, y, c):
    y.append(c.dot(x))
    return y


def main():
    vars = variables()
    x0, x_aim, a1, a2, b, k0, t0, q = vars[0], vars[1], vars[2], vars[3], vars[4], vars[5], vars[6], vars[7]
    u_k = U_k(a1, a2, b, t0, k0, q, x_aim)
    f = F(a1, a2, t0, q)
    print(f'F = {f}')
    g = G(a1, a2, b, t0, q)
    print(f'G = {g}')
    p_f = power_F(k0, a1, a2, t0, q)
    print(f'Пk-1 = {p_f}')
    l, inv_L = L(a1, a2, b, t0, k0, q)
    print(f'L = {l}')
    print(f'L-inverse = {inv_L}')
    l0 = L0(a1, a2, b, t0, k0, q, x_aim)
    print(f'l0 = {l0}')
    c = matrix_C()
    x_k = [x0]
    t = [t0 * i for i in range(k0)]
    for k in range(len(t)+2):
        x_k.append(formula_X(x_k[k], f, g, u_k[k]))
    del x_k[0]

    y = []
    for i in x_k:
        y = formula_Y(i, y, c[0])

    # таблиця
    print(f"0: [0] [0] [0] [{x_k[0][2]}]")
    for i in range(len(x_k)):
        print(f"{str(i+1)}: ", x_k[i][0], x_k[i][1], x_k[i][2], u_k[i])

    list_y = []
    for i in y:
        s = i.tolist()
        list_y.append(s[0])
    list_y1 = []
    for s in range(len(list_y)-2):
        list_y1.append(list_y[s][0])

    list_u = []
    for i in u_k:
        s = i.tolist()
        list_u.append(s[0])
    list_u1 = []
    for s in range(len(list_u)-2):
        list_u1.append(list_u[s][0])

    plt.xlabel('t')
    #plt.ylabel('y(t)')
    plt.ylabel('u(t)')
    plt.xticks(np.arange(t[0], t[-1] + 1, 0.1))
    #plt.plot(t, list_y1, color='pink')
    plt.plot(t, list_u1, color='cyan')
    plt.show()
    return True


main()
