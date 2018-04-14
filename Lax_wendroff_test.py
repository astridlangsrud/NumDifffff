import numpy as np
import matplotlib.pyplot as plt
"""
h = 0.037
k = 0.001
L = 10
x = np.linspace(-L / 2, L / 2, int(L / h) + 1)
sigma = 0.054
tau = 1/120
V_0 = 120
rho_hat = 120
E = 100
c_0 = 54
mu = 600
f_up = 1948
f_rmp = 121
rho_up = 20
N = 500
"""
h = 37.8
k = 0.001
L = 10000
x = np.linspace(-L/2,L/2,int(L/h)+1)
sigma = 56.7
tau = 0.5
V_0 = 2000
rho_hat = 0.14
E = 100
c_0 = 900
mu = 10000
f_up = 32.5
f_rmp = 3.7
rho_up = 0.02
N = 5000

def q(t):
    return 121


def phi(x):
    return 1/(2*np.pi*(sigma**2))*np.exp(-(x ** 2)/(2*(sigma**2)))


def V_ro(ro):
    return V_0 * (1 - (ro / rho_hat)) / (1 + E * ((ro / rho_hat) ** 4))


def s(U, m, n):

    u1 = q(n*k)*phi((m*h)-(L/2))
    u2 = ((V_ro(U[0, m]) - U[1, m])/ tau)
    return np.array([u1, u2])


def s1(U, m, n):
    return np.array([q(n*k)*phi((m*h)-(L/2))])


def s2(U, m, n):
    return np.array([((V_ro(U[0, m]) - U[1, m])/ tau)])


def f_u(U, m):
    u1 = U[0, m] * U[1, m]
    u2 = (1/2)*(U[1, m]**2) + (c_0**2)*np.log(U[0, m])
    return np.array([u1, u2])

def f1_u(U, m):
    return np.array([U[0, m] * U[1, m]])

def f2_u(U, m):
    return np.array([(1/2)*(U[1, m]**2) + (c_0**2)*np.log(U[0, m])])

u = np.zeros([2, len(x) + 1])  # 2 x M
u_next = np.zeros([2, len(x) + 1])
u_half = np.zeros([2, len(x) + 1])
initial_velocity = V_ro(rho_up)
u[0, :] = rho_up
u[1, :] = initial_velocity
u_next[0, :] = rho_up
u_next[1, :] = initial_velocity

for n in range(2):
    s_now = s(u, 1, n)
    f_now = f_u(u, 1)

    for m in range(1, len(x) - 1):
        s_next = s(u, m+1, n)
        f_next = f_u(u, m+1)
        u_half[0, m] = ((u[0, m]+u[0, m+1])-(k/h)*(f_next[0]-f_now[0])+(k/2)*(s_next[0]+s_now[0]))/2
        u_half[1, m] = ((u[1, m]+u[1, m+1])-(k/h)*(f_next[1]-f_now[1])+(k/2)*(s_next[1]+s_now[1]))/2

        u_next[0, m] = u[0, m]-(k/h)*(f1_u(u_half, m)-f1_u(u_half, m-1))+(k/2)*(s1(u_half, m, n)+s1(u_half, m-1, n))
        u_next[1, m] = u[1, m]-(k/h)*(f2_u(u_half, m)-f2_u(u_half, m-1))+(k/2)*(s2(u_half, m, n)+s2(u_half, m-1, n))
        s_now = s_next
        f_now = f_next

    print(u_half)

    u_next[0, 0] = rho_up
    u_next[1, 0] = initial_velocity
    u_next[:, len(x)] = u_next[:, len(x) - 1]
    u = u_next


plt.plot(x, u[0][:-1])
plt.show()