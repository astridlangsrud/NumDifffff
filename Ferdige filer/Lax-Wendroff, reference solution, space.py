import numpy as np
import matplotlib.pyplot as plt
import readwrite as rw


# This script makes a reference solution for Lax-Wendroff in space

# Constants:
h = 4
L = 2**13
k = 10**-4
N = 10**4
x = np.linspace(-L/2,L/2,int(L/h)+1)
sigma = 300
tau = 0.5
V_0 = 2000
rho_hat = 0.14
E = 100
c_0 = 900
mu = 10000
f_up = 32.5
f_rmp = 2
rho_up = 0.02



def q(t):
    return f_rmp


def phi(x):
    return ((2 * np.pi * (sigma ** 2)) ** (-1 /2)) * np.exp(-(x ** 2) / (2 * (sigma ** 2)))


def V_ro(ro):
    return V_0 * (1 - (ro / rho_hat)) / (1 + E * ((ro / rho_hat) ** 4))


def s(U, m, n):
    u1 = q(n * k) * phi((m * h) - (L / 2))
    u2 = ((V_ro(U[0, m]) - U[1, m]) / tau)
    return np.array([u1, u2])


def f_u(U, m):
    u1 = U[0, m] * U[1, m]
    u2 = (1 / 2) * (U[1, m] ** 2) + (c_0 ** 2) * np.log(U[0, m])
    return np.array([u1, u2])

u = np.zeros([2,len(x)+1])
u_next = np.zeros([2,len(x)+1])
u_half = np.zeros([2,len(x)+1])
initial_velocity = V_ro(rho_up)
u_half[0,:] = rho_up
u_half[1,:] = initial_velocity
u[0,:] = rho_up
u[1,:] = initial_velocity
u_next[0,:] = rho_up
u_next[1,:] = initial_velocity

for n in range(N):
    if n % 100 == 0:
        print(int(100 * n / N), "%")
    for m in range(1,len(x)-1):
        s_next = s(u,m,n)
        s_next_p1 = s(u, m+1, n)
        f_next = f_u(u,m)
        f_next_p1 = f_u(u,m+1)

        u_half[0,m] = ((u[0,m] + u[0,m+1])/2) -(k/(2*h))*(f_next_p1[0]-f_next[0])+((k/2)*(s_next[0]+s_next_p1[0]))
        u_half[1,m] = ((u[1,m] + u[1,m+1])/2) -(k/(2*h))*(f_next_p1[1]-f_next[1])+((k/2)*(s_next[1]+s_next_p1[1]))
        f_half_p1 = f_u(u_half,m)
        f_half_m1 = f_u(u_half,m-1)
        s_half_p1 = s(u_half, m, n)
        s_half_m1 = s(u_half, m-1, n)
        u_next[0,m]= u[0,m] -(k/(2*h))*(f_half_p1[0]-f_half_m1[0])+((k/2)*(s_half_p1[0]+s_half_m1[0]))
        u_next[1,m] = u[1,m] -(k/(2*h))*(f_half_p1[1]-f_half_m1[1])+((k/2)*(s_half_p1[1]+s_half_m1[1]))

    u_next[0,0] = u_next[0,1] -(u_next[0,2]-u_next[0,1])
    u_next[1,0] = u_next[1,1] - (u_next[1,2]-u_next[1,1])
    u_next[0, len(x)] = u_next[0, len(x)-1]+(u_next[0,len(x)-1]-u_next[0, len(x)-2])
    u_next[1, len(x)] = u_next[1, len(x) - 1] + (u_next[1, len(x) - 1] - u_next[1, len(x) - 2])
    u = u_next



if __name__ == "__main__":
    rw.write_data(u, "Reference solution, Lax-Wendroff, x-convergence.txt") # Writing the reference solution to file
    a = rw.read_data("Reference solution, Lax-Wendroff, x-convergence.txt")
    print(a)


