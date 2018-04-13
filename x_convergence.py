import numpy as np
import matplotlib.pyplot as plt
import readwrite as rw

"""
h = 0.037
k = 0.0001
L = 10
x = np.linspace(-L/2,L/2,int(L/h)+1)
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
N = 10000

"""

number_of_discretizations = 3
h_values = [500, 1000, 5000]
errors = np.zeros(number_of_discretizations) # The error between the estimated solution and the reference solution


ref_sol = rw.read_data("u_lax_friedrich_x.txt") # Reference solution

L = 10**4
k = 10**-4
sigma = 56.7
tau = 0.5
V_0 = 2000
rho_hat = 0.14
E = 100
c_0 = 900
mu = 10000
f_up = 32.5
f_rmp = 2.166  # 3.7
rho_up = 0.02
N = 10**4


def q(t):
    return f_rmp


def phi(x):
    return ((2 * np.pi * (sigma ** 2)) ** (-1 / 2)) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))


def V_ro(ro):
    return V_0 * (1 - (ro / rho_hat)) / (1 + E * ((ro / rho_hat) ** 4))


def s(U, m, n, h):
    u1 = [0] * (len(m)-1)
    u2 = [0] * (len(m)-1)
    for i in range(1, len(m) - 1):
        u1[i] = q(n * k) * phi((m[i] * h) - (L / 2))
        u2[i] = ((V_ro(U[0, m[i]]) - U[1, m[i]]) / tau)

    return np.array([u1, u2])


def f_u(U, m):
    u1 = U[0, m] * U[1, m]
    u2 = (1 / 2) * (U[1, m] ** 2) + (c_0 ** 2) * np.log(U[0, m])
    return np.array([u1, u2])


def x_convergence(h):
    x = np.linspace(-L / 2, L / 2, int(L/h) + 1)
    u = np.zeros([2, len(x)])  # 2 x M
    u_next = np.zeros([2, len(x)])
    initial_velocity = V_ro(rho_up)
    u[0, :] = rho_up
    u[1, :] = initial_velocity
    u_next[0, :] = rho_up
    u_next[1, :] = initial_velocity
    for n in range(N):
        f = f_u(u, range(1, len(x)))
        s_next = s(u, range(1, len(x)), n, h)

        for m in range(1, len(x) - 2):
            u_next[0, m] = ((u[0, m - 1] + u[0, m + 1]) / 2) - (k / (2 * h)) * (f[0, m + 1] - f[0, m - 1]) + (
                        k * s_next[0, m])
            u_next[1, m] = ((u[1, m - 1] + u[1, m + 1]) / 2) - (k / (2 * h)) * (f[1, m + 1] - f[1, m - 1]) + (
                        k * s_next[1, m]) + ((k / (h ** 2)) * (u[1, m + 1] - 2 * u[1, m] + u[1, m - 1]))
        u_next[0, 0] = rho_up
        u_next[1, 0] = initial_velocity
        u_next[:, len(x)-1] = u_next[:, len(x)-2]
        u = u_next
    error = 0
    print("length of reference:",len(ref_sol[0]))
    print("lenght of u:",len(u[0]))
    relative_discretization = int((len(ref_sol[0])-1)/(len(u[0])-1))
    print("relationship:", relative_discretization)
    for i in range(len(u[0])):
        error += u[0][i]-ref_sol[0][i*relative_discretization]
        print(i/len(u[0]), ((i*relative_discretization)/len(ref_sol[0])))

    return np.sqrt(h)*np.linalg.norm(error)

if __name__ == "__main__":
    for i in range(number_of_discretizations):
        print("h:",h_values[i])
        errors[i] = x_convergence(h_values[i])
        #print(i)
    plt.loglog(h_values, errors)
    plt.xlabel("Step length")
    plt.ylabel("Error")
    plt.show()


