import math
import numpy as np
import matplotlib.pyplot as plt
t0 = 0.001
q = 10
T = 2
n = 2
k0 = np.array([-3/4, 0, 0, 0])
x_est = np.array([1, 1])


def A():
    """Створення матриці А"""
    A = np.zeros((2, 2))
    A[0][1] = -5
    A[1][0] = 0.5
    A[1][1] = -5.4
    return A


def A_est1():
    """Створення матриці А для спостерігача повного порядку"""
    A = np.zeros((2, 2))
    A[0][0] = -1
    A[0][1] = -5
    A[1][0] = 0.5
    A[1][1] = -1
    return A


def A3():
    """Створення матриці А для спостерігача повного порядку"""
    A = np.zeros((1, 1))
    A[0] = -1
    return A


def B():
    """Створення матриці B"""
    B = np.zeros((2, 1))
    B[0][0] = -3
    return B


def B_est1():
    """Створення матриці B для спостерігача повного порядку"""
    B = np.zeros((2, 2))
    B[0][0] = -3
    B[1][0] = 1
    B[1][1] = 1
    return B


def B3():
    """Створення матриці B для спостерігача повного порядку"""
    B = np.zeros((1, 2))
    B[0][0] = -132/19
    B[0][1] = 1
    return B


def C():
    """Створення матриці C"""
    C = np.zeros((1, 2))
    C[0][0] = 1
    return C


def F(x, t, q):
    """Обрахуємо F = e^(A*T0)"""
    I = np.eye(2)  # одинична матриця n*n
    for i in range(1, q + 1):
        I += np.linalg.matrix_power(x.dot(t), i) / math.factorial(i)
    return I


def F2(x, t, q):
    """Обрахуємо F = e^(A*T0)"""
    I = np.eye(1)  # одинична матриця n*n
    for i in range(1, q + 1):
        I += (x * t)**i / math.factorial(i)
    return I


def G(x, q, t, b1):
    """Обрахуємо G = I*B*(T0 + (A*T0^2)/2! + ... + (A^q-1*T0^q)/q!)"""
    I = np.eye(2)
    tmp = t0 * I
    for j in range(2, q):
        tmp += (np.linalg.matrix_power(x, j - 1).dot(I) * (t ** j)) / math.factorial(j)
    return tmp.dot(b1)


def G2(x, q, t, b1):
    """Обрахуємо G = I*B*(T0 + (A*T0^2)/2! + ... + (A^q-1*T0^q)/q!)"""
    I = np.eye(1)
    tmp = t0 * I
    for j in range(2, q):
        tmp += x ** (j - 1) * I * (t ** j) / math.factorial(j)
    return tmp * b1


def formulaX(f, x, u, g):
    """Обрахуємо Xk+1 = F*Xk + G*Uk"""
    res = f.dot(x) + g * u
    return res


def formulaX2(f, x, u, g):
    """Обрахуємо Xk+1 = F*Xk + G*Uk"""
    res = f.dot(x) + g.dot(u)
    return res


def formulaX3(f, x, u, g):
    """Обрахуємо Xk+1 = F*Xk + G*Uk"""
    res = f * x + g.dot(u)
    return res


def X():
    x = np.zeros((2, 1))
    x[0][0] = 5
    x[1][0] = 2
    return x


def X2():
    x = np.zeros((2, 1))
    x[0][0] = 5
    x[1][0] = 2
    return x


def X3():
    x = np.zeros((1, 1))
    x[0][0] = -6/19
    return x


def formulaY(x, c, y):
    """Обрахуємо y = C*x"""
    y.append(c.dot(x))
    return y


def integral(x, u):
    """Інтеграл"""
    res = 0
    if len(u) == 1:
        for i in range(len(np.arange(0, T + t0, t0)) - 1):
            res += t0 * ((0.7 * x[i][0] ** 2 + 0.3 * x[i][1] ** 2 + 5.4 * u[0] ** 2) + (
                        0.7 * x[i + 1][0] ** 2 + 0.3 * x[i + 1][1] ** 2) + 5.4 * u[0] ** 2) / 2
    else:
        for i in range(len(np.arange(0, T + t0, t0)) - 1):
            res += t0 * ((0.7 * x[i][0] ** 2 + 0.3 * x[i][1] ** 2 + 5.4 * u[i] ** 2) + (
                        0.7 * x[i + 1][0] ** 2 + 0.3 * x[i + 1][1] ** 2) + 5.4 * u[i + 1] ** 2) / 2
    return res


def indicator(t, x, u):
    """Індикатор"""
    f1 = integral(x, u) + 3 / 8 * x[-1][0] ** 2
    return f1


def f(k0):
    """Система диференціальних рівнянь, пункт 5"""
    k11, k12, k21, k22 = k0
    k11_d = -0.5*k21 - 0.5*k12 - 5/6*k11**2 + 1.4
    k12_d = -0.5*k22 + 5*k11 + 5.4*k12 - 5/6*k11*k12
    k21_d = 0.5*k11 + 5.4*k21 - 0.5*k22 - 5/6*k11*k21
    k22_d = 5*k12 + 5.4*k22 + 5*k21 + 5.4*k22 + 5/6*k11*k21 + 0.6
    return np.array([k11_d, k12_d, k21_d, k22_d])


def euler_method():
    """Метод Ейлера для розв'язання задачі Коші"""
    num_steps = len(np.arange(0, T + t0, t0))
    k = np.zeros((num_steps, 4))
    k[-1] = k0
    for i in range(num_steps - T, -1, -1):
        derivs = f(k[i + 1])
        k[i] = k[i + 1] - derivs * t0
    k_11 = k[:, 0]
    k_12 = k[:, 1]
    return k_11, k_12


def main():
    vec = X()
    t = list(np.arange(0, T + t0, t0))
    a = A()
    b1 = B()
    f = F(a, t0, q)
    g = G(a, q, t0, b1)
    x = []
    u = [1/2]
    """First"""
    for i in range(len(t)):
        vec = formulaX(f, vec, u[0], g)
        x.append(vec)
    print("First method")
    print("I =", float(indicator(t, x, u)))
    print("x1(2) =", float(x[-1][0]))
    print("x2(2) =", float(x[-1][1]))

    x1 = []
    x2 = []
    for i in range(len(x)):
        x1.append(float(x[i][0]))
    for i in range(len(x)):
        x2.append(float(x[i][1]))
    #plt.plot(t, x1, color='red')
    #plt.plot(t, x2, color='blue')

    """Second"""
    vec = X()
    k_11, k_12 = euler_method()
    u_list = []
    x = [[5, 2]]
    u = -5/18*k_11[0]*x[0][0] - 5/18*k_12[0]*x[0][1]  # использовали первый элемент К и первый элемент Х
    u_list.append(u)  # добавили первый элемент U
    vec = formulaX(f, vec, u_list[0], g)  # посчитали второй элемент Х
    for i in range(1, len(t)):
        x.append(vec)
        """Оптимальний регулятор"""
        U = -5/18*k_11[i]*x[i][0] - 5/18*k_12[i]*x[i][1]
        u_list.append(U)
        vec = formulaX(f, vec, u_list[i], g)
    print("Second method")
    print("I =", float(indicator(t, x, u_list)))
    print("x1(2) =", float(x[-1][0]))
    print("x2(2) =", float(x[-1][1]))

    x1_ = []
    x2_ = []
    u_list_ = []
    for i in range(len(x)):
        x1_.append(float(x[i][0]))
    for i in range(len(x)):
        x2_.append(float(x[i][1]))
    for i in range(len(u_list)):
        u_list_.append(float(u_list[i]))
    #plt.plot(t, x1_, color='red')
    #plt.plot(t, x2_, color='blue')
    #plt.plot(t, u_list_, color='pink')

    """Third"""
    """Повний спостерігач"""
    vec2 = X2()
    x_est = [[5, 2]]
    u_list_est = []
    y_list = []
    t = list(np.arange(0, T + t0, t0))
    a = A_est1()
    b2 = B_est1()
    f1 = F(a, t0, q)
    g1 = G(a, q, t0, b2)
    c = C()
    for i in range(0, len(t)):
        """Оптимальний регулятор"""
        y = formulaY(x[i], c, y_list)
        U = -5 / 18 * k_11[i] * x_est[i][0] - 5 / 18 * k_12[i] * x_est[i][1]
        u_list_est.append(U)
        vec_u = np.zeros((2, 1))
        vec_u[0][0] = U
        vec_u[1][0] = y[i].flatten()[0]
        vec2 = formulaX2(f1, vec2, vec_u, g1)
        x_est.append(vec2)
    print("Third method, спостерігач повного порядку")
    print("I =", float(indicator(t, x_est, u_list_est)))
    print("x1(2) =", float(x_est[-1][0]))
    print("x2(2) =", float(x_est[-1][1]))

    x1_list = []
    x2_list = []
    x1_list_est = []
    x2_list_est = []
    u_list_est_ = []
    for i in range(len(x)):
        x1_list.append(float(x[i][0]))
    for i in range(len(x)):
        x2_list.append(float(x[i][1]))
    for i in range(len(x_est) - 1):
        x1_list_est.append(float(x_est[i][0]))
    for i in range(len(x_est) - 1):
        x2_list_est.append(float(x_est[i][1]))
    for i in range(len(u_list_est)):
        u_list_est_.append(float(u_list_est[i]))

    #plt.plot(t, x1_list, color='red')
    #plt.plot(t, x2_list, color='blue')
    #plt.plot(t, x1_list_est, color='green')
    #plt.plot(t, x2_list_est, color='yellow')
    #plt.plot(t, u_list_est_, color='pink')

    """Нижній спостерігач"""
    vec3 = X3()
    a3 = A3()
    b3 = B3()
    u_list_est2 = []
    z_list = [-6 / 19]
    f2 = F2(a3, t0, q)
    g2 = G2(a3, q, t0, b3)
    x1est = []
    x2est = []
    for i in range(0, len(t)):
        y = formulaY(x[i], c, y_list)
        x1est.append(y[i].flatten()[0])
        U2 = (-5 / 18 * k_11[i] - 5 / 18 * k_12[i]) * z_list[i]
        vec_u2 = np.zeros((2, 1))
        vec_u2[0][0] = U2
        vec_u2[1][0] = y[i].flatten()[0]
        vec3 = formulaX3(f2, vec3, vec_u2, g2)
        z_list.append(vec3[0][0])
        x2est.append(22/25 * x1est[i] + 19/44 * z_list[i])
    for i in range(0, len(t)):
        U = -5 / 18 * k_11[i] * x1est[i] - 5 / 18 * k_12[i] * x2est[i]
        u_list_est2.append(U)
    final = list(zip(x1est, x2est))
    print("Third method, спостерігач нижнього порядку")
    print("I =", float(indicator(t, final, u_list_est2)))
    print("x1(2) =", float(final[-1][0]))
    print("x2(2) =", float(final[-1][1]))

    """Графік"""
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.grid()
    plt.xticks(np.arange(t[0], t[-1] + 1, 20))
    x1_lst = []
    x2_lst = []
    x1est_ = []
    x2est_ = []
    u_list_est2_ = []
    for i in range(len(x)):
        x1_lst.append(float(x[i][0]))
    for i in range(len(x)):
        x2_lst.append(float(x[i][1]))
    for i in range(len(x1est)):
        x1est_.append(float(x1est[i]))
    for i in range(len(x2est)):
        x2est_.append(float(x2est[i]))
    for i in range(len(u_list_est2)):
        u_list_est2_.append(float(u_list_est2[i]))
    #plt.plot(t, x1_lst, color='red')
    #plt.plot(t, x2_lst, color='blue')
    #plt.plot(t, x1est_, color='green')
    #plt.plot(t, x2est_, color='yellow')
    #plt.plot(t, u_list_est2_, color='pink')
    plt.show()


main()
