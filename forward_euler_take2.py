import numpy as np
import matplotlib.pyplot as plt

h = 0.037
k = 0.0001
L = 10
x = np.linspace(-L/2,L/2,int(L/h)+1)
sigma = 0.057
tau = 1/120
V_0 = 120
rho_hat = 120
E = 100
c_0 = 54
mu = 600
f_up = 1857
f_rmp = 121
rho_up = 20

def V_ro(ro):
    return V_0*(1-(ro/rho_hat))/(1+E*((ro/rho_hat)**4))

def q(t):
    return 121

def phi(x):
    return 2*np.pi*(sigma**2)*np.exp(-(x**2)/(2*(sigma**2)))

def b(U,n):

    u1 = q(n*k)*phi(m*h)
    u2 = ((V_ro(U[0,m])-U[1,m])/tau)-(((c_0**2)*((U[0,m+1]-U[0,m])/h))-mu*(U[1,m+1]-2*U[1,m]+U[1,m-1])/(h*h))/U[0,m]
    return np.array([u1, u2])
"""
u = np.zeros([2,len(x)]) #2 x M
u_next = np.zeros([2,len(x)])
initial_velocity = V_ro(rho_up)
u[0,0] = rho_up
u[1,0] = initial_velocity
"""