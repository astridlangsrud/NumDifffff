import numpy as np
import matplotlib.pyplot as plt
h = 37.8
k = 0.001
L = 10000
x = np.linspace(-L/2,L/2,int(L/h)+1)
sigma = 56.7
tau = 30
V_0 = 120/3.6
rho_hat = 0.14
E = 100
c_0 = 54/3.6
mu = 600/3.6
f_up = 2000/3600
f_rmp = 130/3600
rho_up = 0.02
N = 30000

def phi(x):
    return ((2*np.pi*(sigma**2))**(-1/2))*np.exp(-(x**2)/(2*(sigma**2)))

def q(t):
    return 1#f_rmp

def V_ro(ro):
    return V_0*(1-(ro/rho_hat))/(1+E*((ro/rho_hat)**4))

def s(U,n):
    s1 = [0]*(len(x))
    s2 = [0]*(len(x))
    for i in range(len(x)):
        s1[i] = q(n*k)*phi((i*h)-(L/2))
        s2[i] = ((V_ro(U[0,i]) - U[1,i])/tau)
    return np.array([s1, s2])

initial_velocity = V_ro(rho_up)

u = np.zeros([2,len(x)]) #2 x M
u_next = np.zeros([2,len(x)])

u[0,:] = rho_up
u[1,:] = initial_velocity
u_next[0,:] = rho_up
u_next[1,:] = initial_velocity

for n in range(N):
    S = s(u, n)
    for m in range(len(u)-1):
        ro_x = (u[0,m+1]-u[0,m])/h
        nu_x = (u[1,m+1]-u[0,m])/h
        u_next[0,m] = u[0,m] - (k/(h))*(ro_x*u[1,m]+nu_x*u[0,m]) + k*S[0,m]
        u_next[1,m] = u[1,m] - (k/h)*(nu_x*u[1,m] + c_0*c_0*ro_x/u[0,m]) + k*S[1,m]
    u_next[0,-1] = (2*u_next[0,-2]-u_next[0,-3])
    u_next[1, -1] = (2 * u_next[0, -2] - u_next[0, -3])
    u = u_next
    if n%10000 == 0:
        print(n)
    #    plt.plot(x, u[0])
    #    #plt.plot(x, u[1])
    #    plt.show()

plt.plot(x, u[0])

plt.show()


