import numpy as np
import matplotlib.pyplot as plt

h = 5#0.037
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
f_rmp = 121
rho_up = 20
N = 4

def q(t):
    return 121*100

def phi(x):
    return 2*np.pi*(sigma**2)*np.exp(-(x**2)/(2*(sigma**2)))

def V_ro(ro):
    return V_0*(1-(ro/rho_hat))/(1+E*((ro/rho_hat)**4))

def s(U,m,n):
    u1 = q(n*k)*phi(m*h)
    u2 = ((V_ro(U[0,m])-U[1,m])/tau)
    return np.array([u1, u2])

def f_u(U,m):
    u1 = U[0,m]*U[1,m]
    u2 = 1/2*(U[1,m]**2) + (c_0**2)*np.log(U[0,m])
    return np.array([u1,u2])

U = np.zeros([2,N,len(x)])
initial_velocity = V_ro(rho_up)
U[0,0,:] = rho_up
U[1,0,:] = initial_velocity
print(U)