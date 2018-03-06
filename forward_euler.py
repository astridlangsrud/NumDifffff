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
f_up = 1948
f_rmp = 121
rho_up = 20
#bruk u' = u_m - u_(m-1) / h
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
    print(((U[0,m]-U[0,m-1])/h)*U[1,m])
    u1 = ((U[0,m]-U[0,m-1])/h)*U[1,m]+((U[1,m+1]-U[1,m])/h)*U[0,m]
    u2 = U[1,m]*((U[1,m+1]-U[1,m])/h)
    return np.array([u1,u2])


u = np.zeros([2,len(x)]) #2 x M
u_next = np.zeros([2,len(x)])
initial_velocity = V_ro(rho_up)
u[0,0] = rho_up
u[1,0] = initial_velocity

for n in range(10):
    u_next[0, 0] = rho_up
    u_next[1, 0] = initial_velocity
    for m in range(len(x)-1):
        b_next = b(u,m,n)
        f_next = f_u(u,m)
        u_next[0,m] = u[0,m]+k*(b_next[0]-f_next[0])
        u_next[1, m] = u[1,m]+k*(b_next[1]-f_next[1])
        u = u_next



#plt.plot(x,q(0)*phi(x))
#plt.show()