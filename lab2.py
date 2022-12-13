import numpy as np
import matplotlib.pyplot as plt
import math


def A(a1, a2):
    """Створення матриці А з заданими параметрами а1 та а2"""
    a = np.zeros((3, 3))
    a[0][1] = 1
    a[1][2] = 1
    a[2][0] = -1
    a[2][1] = -a1
    a[2][2] = -a2
    return a


def B(b1):
    """Створення матриці B з заданим параметром b = 1"""
    b = np.zeros((3, 1))
    b[0][0] = 0
    b[1][0] = 0
    b[2][0] = b1
    return b


def C():
    """Створення матриці C"""
    c = []
    for i in range(3):
        value = np.zeros((1, 3))
    if i != 0:
        value[0][i] = 0
    else:
        value[0][i] = 1
    c.append(value)
    return c


def F(a, t, q):
    """Обрахуємо F = e^(A*T0)"""
    I = np.eye(3)  # одинична матриця n*n
    for i in range(1, q + 1):
        I += np.linalg.matrix_power(a.dot(t), i) / math.factorial(i)
    print("F", I)
    return I


def G(a, q, t, b):
    """Обрахуємо G = I*B*(T0 + (A*T0^2)/2! + ... + (A^q-1*T0^q)/q!)"""
    b1 = B(b)
    I = np.eye(3)
    tmp = t * I
    for j in range(2, q):
        tmp += (np.linalg.matrix_power(a, j - 1).dot(I) * (t ** j)) / math.factorial(j)
    print("G", tmp.dot(b1))
    return tmp.dot(b1)


u = 1


def formula(f, x, g, l):
    """Обрахуємо Xk+1 = F*Xk + G*Uk, de Uk = -l(transpose)*Xk+U`k, U`k = 1"""
    u_k = -l.transpose().dot(x) + u
    res = f.dot(x) + g.dot(u_k)
    return res


def formula_2(x, c, y):
    """Обрахуємо y = C*x"""
    y.append(c.dot(x))
    return y


def variables():
    """Задаємо параметри"""
    a1 = float(input('Введіть a1 для матриці A: '))
    a2 = float(input('Введіть a2 для матриці A: '))
    b1 = float(input('Введіть b для матриці B: '))
    t0 = float(input('Введіть t0: '))
    q = int(input('Введіть q: '))
    var = [a1, a2, t0, q, b1]
    return var


def J(x, t0):
    sum = 0.0
    for i in range(len(x)):
        sum += abs(x[i][0][0] - 1) * t0
    return sum


delta_l = 0.05 # крок


def optimisation(f, g, x, t, l, t0, step):
    """Процес оптимізації"""
    # Початкове J
    x_j = [x]
    for i in range(len(t)):
        vec = formula(f, x_j[i], g, l)
        x_j.append(vec)
    j = J(x_j, t0)
    print('Початкова J: ', j)
    # Наступне J
    flag = False
    l[step][0] += delta_l
    x_j_i = [x]
    for i in range(len(t)):
        vec = formula(f, x_j_i[i], g, l)
        x_j_i.append(vec)
    j_i = J(x_j_i, t0)
    while j - j_i > 0:
        j = j_i
    x_j_i = [x]
    l[step][0] += delta_l
    for i in range(len(t)):
        vec = formula(f, x_j_i[i], g, l)
        x_j_i.append(vec)
    j_i = J(x_j_i, t0)
    flag = True
    while j - j_i <= 0 and flag is False:
        j = j_i
    x_j_i = [x]
    l[step][0] -= delta_l
    for i in range(len(t)):
        vec = formula(f, x_j_i[i], g, l)
        x_j_i.append(vec)
    j_i = J(x_j_i, t0)
    print('Оптимальна J: ', abs(l[step][0]))
    return x_j_i


def main():
    vars = variables()
    a = A(vars[0], vars[1])
    f = F(a, vars[2], vars[3])
    g = G(a, vars[3], vars[2], vars[4])
    c = C()
    l = np.array([[0.0], [0.0], [0.0]])  # l2 =0, l3 = 0
    T = list(np.arange(0, 40 + vars[2], vars[2]))
    I = np.zeros((3, 1))
    # Запит варіанту
    option = int(input("Опція: "))
    x1 = []
    x1.append(I)
    for i in range(len(T)):
        vec = formula(f, x1[i], g, l)
        x1.append(vec)
    l[option][0] = 0.0
    x1_opt = optimisation(f, g, I, T, l, vars[2], option)
    y = []
    y_opt = []
    for i in x1:
        y = formula_2(i, c[0], y)
    for i in x1_opt:
        y_opt = formula_2(i, c[0], y_opt)
    lst = []
    for i in y:
        a = i.tolist()
        lst.append(a[0])
    list_y = []
    for s in range(len(lst) - 1):
        list_y.append(lst[s][0])
    lst1 = []
    for i in y_opt:
        a = i.tolist()
        lst1.append(a[0])
    list_y_opt = []
    for s in range(len(lst1) - 1):
        list_y_opt.append(lst1[s][0])
    # Графік
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.grid()
    plt.xticks(np.arange(T[0], T[-1] + 1, 10))
    plt.plot(T, list_y, label='not optimal', color='cyan')
    plt.plot(T, list_y_opt, label='optimal', color='pink')
    plt.legend()
    plt.show()


main()
