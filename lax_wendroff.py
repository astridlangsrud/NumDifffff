import numpy as np
import matplotlib.pyplot as plt

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

u = np.zeros([2,len(x)+1]) #2 x M
u_next = np.zeros([2,len(x)+1])
initial_velocity = V_ro(rho_up)
u[0,:] = rho_up
u[1,:] = initial_velocity
u_next[0,:] = rho_up
u_next[1,:] = initial_velocity
for n in range(N):
    #u[:,len(x)] = u[:,len(x)-1]
    #u[:,0] = u[:,1]
    for m in range(1,len(x)-1):
        s_next = s(u,m,n)
        f_next_m1 = f_u(u,m-1) #m-1
        f_next_p1 = f_u(u,m+1) #m+1
        u_next[0,m] = ((u[0,m-1]+ u[0,m+1])/2) -(k/(2*h))*(f_next_p1[0]-f_next_m1[0])+(k*s_next[0])
        u_next[1,m] = ((u[1,m-1]+ u[1,m+1])/2) -(k/(2*h))*(f_next_p1[1]-f_next_m1[1])+(k*s_next[1])

        
        u_next[0,m] = ((u[0,m-1]+ u[0,m+1])/2) -(k/(2*h))*(f_next_p1[0]-f_next_m1[0])+(k*s_next[0])
        u_next[1,m] = ((u[1,m-1]+ u[1,m+1])/2) -(k/(2*h))*(f_next_p1[1]-f_next_m1[1])+(k*s_next[1])
        
        #print(((u[0, m - 1] + u[0, m + 1]) / 2), (k / (2 * h)) * (f_next_p1[0] - f_next_m1[0]), (k * s_next[0]))
        #print(((u[1,m-1]+ u[1,m+1])/2),(k/(2*h))*(f_next_p1[1]-f_next_m1[1]),(k*s_next[1]))
    u_next[0,0] = rho_up
    u_next[1,0] = initial_velocity
    u_next[:, len(x)] = u_next[:, len(x) - 1]
    u = u_next
    #u[:, len(x)] = u[:, len(x) - 1]
    #u[:, 0] = u[:, 1]

plt.plot(x,u[0][:-1])
plt.show()