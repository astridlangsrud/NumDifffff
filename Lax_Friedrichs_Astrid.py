import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
h = 37.8
k = 0.0001
L = 10000
=======
h = 0.37
k = 0.01
L = 10

>>>>>>> 236ee7c07f148d05bdebc1faa3060b7563a9bb21
x = np.linspace(-L/2,L/2,int(L/h)+1)
sigma = 56.7
tau = 0.5
V_0 = 2000
rho_hat = 0.14
E = 100
<<<<<<< HEAD
c_0 = 900
mu = 10000
f_up = 32.5
f_rmp = 2
rho_up = 0.02
N = 100000
=======
c_0 = 54
mu = 600
f_up = 1948
f_rmp = 121
rho_up = 20
<<<<<<< Updated upstream
N = 100
<<<<<<< HEAD
>>>>>>> 236ee7c07f148d05bdebc1faa3060b7563a9bb21
=======
=======
<<<<<<< Updated upstream
N = 10000
=======
N = 1000
>>>>>>> Stashed changes
>>>>>>> Stashed changes
>>>>>>> 9936aba758aee1d78137d45cada94c117d18d268

def q(t):
    return f_rmp*100

def phi(x):
    return ((2*np.pi*(sigma**2))**(-1/2))*np.exp(-(x**2)/(2*(sigma**2)))

def V_ro(ro):
    print(ro)
    return V_0*(1-(ro/rho_hat))/(1+E*((ro/rho_hat)**4))

def s(U,m,n):
<<<<<<< Updated upstream
    u1 = q(n*k)*phi((m*h)-(L/2))
<<<<<<< HEAD
<<<<<<< HEAD
=======
    #print(phi((m*h)-(L/2)))
>>>>>>> 236ee7c07f148d05bdebc1faa3060b7563a9bb21
=======
<<<<<<< Updated upstream
    #print(phi((m*h)-(L/2)))
=======
    print(phi((m*h)-(L/2)))
=======
    u1 = q(n*k)*phi(m*h-L/2)
>>>>>>> Stashed changes
>>>>>>> Stashed changes
>>>>>>> 9936aba758aee1d78137d45cada94c117d18d268
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
    #print(n)
    #u[:,len(x)] = u[:,len(x)-1]
    #u[:,0] = u[:,1]
    for m in range(1,len(x)-1):
        s_next = s(u,m,n)
        f_next_m1 = f_u(u,m-1) #m-1
        f_next_p1 = f_u(u,m+1) #m+1
        u_next[0,m] = ((u[0,m-1]+ u[0,m+1])/2) -(k/(2*h))*(f_next_p1[0]-f_next_m1[0])+(k*s_next[0])
        u_next[1,m] = ((u[1,m-1]+ u[1,m+1])/2) -(k/(2*h))*(f_next_p1[1]-f_next_m1[1])+(k*s_next[1]) # + (((k/(h**2))*(u[1,m+1]-2*u[1,m]+u[1,m-1])))
        #print(((u[0, m - 1] + u[0, m + 1]) / 2), (k / (2 * h)) * (f_next_p1[0] - f_next_m1[0]), (k * s_next[0]))
        #print(((u[1,m-1]+ u[1,m+1])/2),(k/(2*h))*(f_next_p1[1]-f_next_m1[1]),(k*s_next[1]))
        #Legge til at koden skal stoppe å kjøre hvis ro går over ro_hat
    u_next[0,0] = rho_up
    u_next[1,0] = initial_velocity
    u_next[:, len(x)] = u_next[:, len(x) - 1]
    u = u_next
    #u[:, len(x)] = u[:, len(x) - 1]
    #u[:, 0] = u[:, 1]

plt.plot(x,u[0][:-1])
plt.show()