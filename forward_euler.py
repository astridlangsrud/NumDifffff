import numpy as np
import matplotlib.pyplot as plt

h = 0.0037 # step-length in x-direction
k = 10**(-7) # time-step
N = 100 # number of time-steps
L = 10 # length of highway
x = np.linspace(-L/2, L/2, int(L/h)+1) # points along the highway
sigma = 0.054
tau = 1/120
V_0 = 120 # speed limit
rho_hat = 120 # maximum density
E = 100
c_0 = 54
mu = 600
f_up = 1948 # car-flux before the ramp
f_rmp = 121 # car-flux on the ramp
rho_up = 20 # density before the ramp


def q(t):
    return f_rmp

def phi(x):
    return 2*np.pi*(sigma**2)*np.exp(-(x**2)/(2*(sigma**2)))

def V_ro(ro):
    return V_0*(1-(ro/rho_hat))/(1+E*((ro/rho_hat)**4))

def b(U,m,n):
    u1 = q(n*k)*phi(m*h)
    u2 = ((V_ro(U[0,m])-U[1,m])/tau)-(((c_0**2)*((U[0,m+1]-U[0,m])/h))-mu*(U[1,m+1]-2*U[1,m]+U[1,m-1])/(h**2))/U[0,m]
    return np.array([u1, u2])

def f_u(U,m):
    u1 = ((U[0,m+1]-U[0,m])/h)*U[1,m]+((U[1,m+1]-U[1,m])/h)*U[0,m] # rho'*v + v'*rho
    u2 = U[1,m]*((U[1,m+1]-U[1,m])/h) # v*v'
    return np.array([u1,u2])

def forward_euler(k, N):
    u = np.zeros([2, len(x)]) # 2 x M
    u_next = np.zeros([2,len(x)])
    initial_velocity = V_ro(rho_up)
    u[0,:] = rho_up
    u[1,:] = initial_velocity

    for n in range(N):
        for m in range(len(x)-1):
            b_next = b(u,m,n)
            f_next = f_u(u,m)
            u_next[0,m] = u[0,m]+k*(b_next[0]-f_next[0])
            u_next[1, m] = u[1,m]+k*(b_next[1]-f_next[1])
        #print(u[0,])

        u = u_next
    #print(u)
    #print(len(u[0]))
    return u


if __name__ == "__main__":
    u = forward_euler(k, N)
    for i in range(len(u[0])):
        if u[0,i] < 10:
            print(i)
    #print(len(x[:-1]))
    plt.plot(x[:-1], u[0][:-1])
    plt.show()
