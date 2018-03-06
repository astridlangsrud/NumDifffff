import numpy as np
import matplotlib.pyplot as plt

h = 0.5 # step-length in x-direction
k = 0.01 # step-length in time
L = 10 # length of highway
x = np.linspace(-L/2,L/2,int(L/h)+1) # points along the highway
sigma = 0.054
tau = 1/120
V_0 = 120 # speed limit
rho_hat = 120 # maximum density
E = 100
c_0 = 54 #
mu = 600
f_up = 1948 # car-flux before the ramp
f_rmp = 121 # car-flux on the ramp
rho_up = 20 # density before the ramp


def q(t):
    return 121

def phi(x):
    return 2*np.pi*(sigma**2)*np.exp(-(x**2)/(2*(sigma**2)))

def V_ro(ro):
    return V_0*(1-(ro/rho_hat))/(1+E*((ro/rho_hat)**4))

def b(U,m,n):
    u1 = q(n*k)*phi(m*h)
    u2 = ((V_ro(U[0,m])-U[1,m])/tau)-(((c_0**2)*((U[0,m+1]-U[0,m])/h))-mu*(U[1,m+1]-2*U[1,m]+U[1,m-1])/(h*h))/U[0,m]
    return np.array([u1, u2])

def f_u(U,m):
    u1 = ((U[0,m+1]-U[0,m])/h)*U[1,m]+((U[1,m+1]-U[1,m])/h)*U[0,m]
    u2 = U[1,m]*((U[1,m+1]-U[1,m])/h)
    return np.array([u1,u2])

def forward_euler(h, k):
    u = np.zeros([2, len(x)]) # 2 x M
    u_next = np.zeros([2,len(x)])
    initial_velocity = V_ro(rho_up)
    u[0,:] = rho_up
    u[1,:] = initial_velocity
for n in range(10):
    for m in range(len(x)-1):
        b_next = b(u,m,n)
        f_next = f_u(u,m)
        print(u[0,1:10])
        u_next[0,m] = u[0,m]+k*(b_next[0]-f_next[0])
        u_next[1, m] = u[1,m]+k*(b_next[1]-f_next[1])
        u = u_next
print(u)