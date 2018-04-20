import numpy as np
import matplotlib.pyplot as plt
import readwrite as rw

# This script makes a convergence plot for Lax-Friedrich in space

# Constants:
k = 10**-4
L = 2**13
sigma = 300
tau = 0.5
V_0 = 2000
rho_hat = 0.14
E = 100
c_0 = 900
mu = 100
f_up = 32.5
f_rmp = 2
rho_up = 0.02
N = 10**4

# The different step lengths we are iterating for
h_values = [2**i for i in range(2, 9)]

number_of_discretizations = len(h_values)
errors2 = np.zeros(number_of_discretizations) # Error in 2-norm

ref_sol = rw.read_data("Reference solution, Lax-Friedrich, x-convergence.txt") # Reference solution
ref_sol_points = L/(len(ref_sol[0])-1) # Relationship between length of the road and length of reference solution




def q(t):
    return f_rmp

def phi(x):
    return ((2*np.pi*(sigma**2))**(-1/2))*np.exp(-(x**2)/(2*(sigma**2)))

def V_ro(ro):
    return V_0*(1-(ro/rho_hat))/(1+E*((ro/rho_hat)**4))

def s(U,m,n,h):
    u1 = q(n*k)*phi((m*h)-(L/2))
    u2 = ((V_ro(U[0,m])-U[1,m])/tau) + ((k/(h**2))*(U[1,m+1]-2*U[1,m]+U[1,m-1]))*(mu/U[0,m])
    return np.array([u1, u2])

def f_u(U,m):
    u1 = U[0,m]*U[1,m]
    u2 = 1/2*(U[1,m]**2) + (c_0**2)*np.log(U[0,m])
    return np.array([u1,u2])

def x_convergence(h): # Returns the error in the 2-norm for step length h
    x = np.linspace(-L/2, L/2, int(L/h) + 1)
    u = np.zeros([2, len(x) + 1]) #2 x M
    u_next = np.zeros([2, len(x) + 1])
    initial_velocity = V_ro(rho_up)
    u[0,:] = rho_up
    u[1,:] = initial_velocity
    u_next[0,:] = rho_up
    u_next[1,:] = initial_velocity
    for n in range(N):
        for m in range(1, len(x)):
            s_next = s(u, m, n, h)
            fm = f_u(u, m - 1)
            fp = f_u(u, m + 1)
            u_next[0, m] = ((u[0, m-1] + u[0, m+1])/2) - (k/(2*h))*(fp[0]-fm[0])+(k*s_next[0])
            u_next[1, m] = ((u[1, m-1] + u[1, m+1])/2) - (k/(2*h))*(fp[1]-fm[1])+(k*s_next[1]) + ((k/(h**2))*(u[1, m+1]-2*u[1, m]+u[1, m-1]))*(mu/u[0, m])
        u_next[0,0] = rho_up
        u_next[1,0] = initial_velocity
        u_next[0, len(x)] = u_next[0, len(x)-1]+(u_next[0,len(x)-1]-u_next[0, len(x)-2])
        u_next[1, len(x)] = u_next[1, len(x)-1]+(u_next[1,len(x)-1]-u_next[1, len(x)-2])
        u = u_next

    error = np.zeros(len(ref_sol[0])-1) # error[i] contains the difference between # ref_sol[i] and u[i]

    for i in range(1, len(ref_sol[0])-1):
        error[i] = np.interp(ref_sol_points*i-L/2, x, u[0,:-1]) - ref_sol[0,i] # The first term is the interpolation

    return np.sqrt(h)*np.linalg.norm(ord = 2, x = error) # The 2-norm of the error



if __name__ == "__main__":
    for i in range(number_of_discretizations):
        errors2[i] = x_convergence(h_values[i])

    plt.loglog(h_values, errors2, "b")
    plt.loglog(h_values, [errors2[0] / ((h_values[0])) * (h_values[i]) for i in range(len(h_values))], "r--")
    # Scaling the 1st order reference in order to make it intercept with the error curve in the starting point
    plt.xlabel("h")
    plt.legend(["Error in 2-norm", "1st order reference"])
    plt.ylabel("Error")
    plt.show()
