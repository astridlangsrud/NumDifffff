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
k = 0.0001
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
N = 10000

def q(t):
    return 121


def phi(x):
    return 1/(2*np.pi*(sigma**2))*np.exp(-(x ** 2)/(2*(sigma**2)))


def V_ro(ro):
    return V_0 * (1 - (ro / rho_hat)) / (1 + E * ((ro / rho_hat) ** 4))


def s(U, m, n):
    u1 = [0]*(len(x)-1)
    u2 = [0]*(len(x)-1)
    print(len(x))
    #print(len(u1))
    for i in range(1,len(x)-1):
        #print(phi(m[i]*h-L/2))
        #print(i)
        u1[i] = q(n*k)*phi((m[i]*h)-(L/2))
        u2[i] = ((V_ro(U[0, m[i]]) - U[1, m[i]])/tau)
    return np.array([u1, u2])


def f_u(U, m):
    u1 = U[0, m] * U[1, m]
    u2 = (1/2)*(U[1, m]**2) + (c_0**2)*np.log(U[0, m])
    return np.array([u1, u2])


#m_vec = np.linspace(1, len(x)-1, len(x)-1)

u = np.zeros([2, len(x) + 1])  # 2 x M
u_next = np.zeros([2, len(x) + 1])
u_half = np.zeros([2, len(x) + 1])
initial_velocity = V_ro(rho_up)
u[0, :] = rho_up
u[1, :] = initial_velocity
u_next[0, :] = rho_up
u_next[1, :] = initial_velocity

for n in range(N):
    s_next = s(u, range(1,len(x)+1), n)
    f_next = f_u(u, range(1,len(x)+1))
    for m in range(1, len(x) - 1):
        u_half[0, m] = ((u[0, m]+u[0, m+1])-(k/h)*(f_next[0][m+1]-f_next[0][m])+(k/2)*(s_next[0][m]+s_next[0][m+1]))/2
        u_half[1, m] = ((u[1, m]+u[1, m+1])-(k/h)*(f_next[1][m+1]-f_next[1][m])+(k/2)*(s_next[1][m]+s_next[1][m+1]))/2

        u_next[0, m] = u[0, m]-(k/h)*(f_u(u_half, m)[0]-f_u(u_half, m-1)[0])+(k/2)*(s(u_half, m, n)[0]+s(u_half, m-1, n)[0])
        u_next[1, m] = u[1, m]-(k/h)*(f_u(u_half, m)[1]-f_u(u_half, m-1)[1])+(k/2)*(s(u_half, m, n)[1]+s(u_half, m-1, n)[1])

    u_next[0, 0] = rho_up
    u_next[1, 0] = initial_velocity
    u_next[:, len(x)] = u_next[:, len(x) - 1]
    u = u_next
    # u[:, len(x)] = u[:, len(x) - 1]
    # u[:, 0] = u[:, 1]

plt.plot(x, u[0][:-1])
plt.show()