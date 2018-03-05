
import matplotlib.pyplot as plt
import numpy as np
"""
# Task 1:

t = 0.5 # We want the numerical solution u(x,0.5)
h = 0.01 # Length of steps in x-direction
x_min = 0 # 0<x<1
x_max = 1
x_points = int(1+(x_max-x_min)/h) # Number of points in x-direction (x = 0 and x = 1 included)


# Case i:
p_i = -1.5 # p = -k/h

# Case ii:
p_ii = -0.5

# Case iii:
p_iii = -0.25

x_vector = np.linspace(x_min, x_max, x_points) # Included x = 1
print("Which p do you want? (i: p = -1.5, ii: p = -0.5, iii: p = -0.25)")
variable = input()
if variable == "i":
    p = p_i
elif variable == "ii":
    p = p_ii
elif variable == "iii":
    p = p_iii
else:
    print("wrong input")


k = -p*h # Time steps
time_points = int(t/k+1) # Number of time steps


def analytic_solution(x, t):
    if x < 1-t:
        return np.sin(2*np.pi*(x+t))
    else:
        return 0

def make_analytic(k):
    u = np.zeros((time_points, x_points))
    for n in range(time_points):
        for m in range(x_points):
            u[n][m] = analytic_solution(h*m, k*n) # u(x, t) for x = 0, h, ... 1, and t = 0, k, ... 0.5
    return u


def Lax_Friedrichs():
    u = make_analytic(k) # Analytic solution
    U = np.zeros((time_points, x_points)) # Numerical solution
    for m in range(x_points):
        U[0][m] = analytic_solution(m*h, 0) # Initial value condition: u(x, 0) = sin(2*pi*x)
    for n in range(time_points-1): # t =  0.01, 0.02, ... 0.5
        for m in range(1, x_points-1): # u(x,t) for h < x < 1-h
            U[n+1][m] = 0.5*(U[n][m+1]+U[n][m-1])+(k/(2*h))*(U[n][m+1]-U[n][m-1])

        U[n+1][0] = -U[n+1][1]+2*U[n+1][2] # u(0,t) = -u(h,t)+2u(2h,t)
        U[n+1][-1] = 0 # u(1,t) = 0

    return u, U

analytic, numerical = Lax_Friedrichs()

plt.plot(x_vector, analytic[-1])
plt.plot(x_vector, numerical[-1])
plt.legend(["analytic", "numerical"])
plt.title("(i): p=-1.5")
plt.show()
"""

#"""
# Task 2:

t = 0.5 # We want the numerical solution u(x,0.5)
h = 0.05 # Length of steps in x-direction
x_min = 0 # 0<x<1
x_max = 1
x_points = int(1+(x_max-x_min)/h) # Number of points in x-direction (x = 0 and x = 1 included)
a = -1

# Case i:
p = - 1/3 # p = -k/h

x_vector = np.linspace(x_min, x_max, x_points) # Included x = 1

k = -p*h # Time steps
time_points = int(t/k) # Number of time steps


def analytic_solution(x, t):
    if x < 1-t and x > 0.5-t:
        return 1
    else:
        return 0

def make_analytic(k):
    u = np.zeros((time_points, x_points))
    for n in range(time_points):
        for m in range(x_points):
            u[n][m] = analytic_solution(h*m, k*n) # u(x, t) for x = 0, h, ... 1, and t = 0, k, ... 0.5
    return u


def upwind_scheme():
    u = make_analytic(k) # Analytic solution
    U = np.zeros((time_points, x_points)) # Numerical solution
    for m in range(x_points):
        U[0][m] = analytic_solution(m*h, 0) # Initial value condition: u(x,0) = 1 for x > 0.5 and u(x,0) = 0 for x < 0.5
    for n in range(time_points-1): # t =  0.01, 0.02, ... 0.5
        for m in range(x_points-1): # u(x,t) for h < x < 1-h
            U[n+1][m] = U[n][m]+(k/h)*(U[n][m+1]-U[n][m])

        U[n+1][-1] = 0 # u(1,t) = 0

    return u, U


def Lax_Wendroff():
    U = np.zeros((time_points, x_points)) # Numerical solution
    for m in range(x_points):
        U[0][m] = analytic_solution(m*h, 0) # Initial value condition: u(x, 0) = sin(2*pi*x)
    for n in range(time_points-1): # t =  0.01, 0.02, ... 0.5
        for m in range(1, x_points-1): # u(x,t) for h < x < 1-h
            U[n+1][m] = U[n][m]-0.5*(a*k/h)*(U[n][m+1]-U[n][m-1])+0.5*((a*k/h)**2)*(U[n][m+1]-2*U[n][m]+U[n][m-1])

        U[n+1][0] = U[n+1][1] # u(0,t) = u(h,t)
        U[n+1][-1] = 0 # u(1,t) = 0

    return U # Only needs to return the numerical solution, since the upwind scheme returns the analytic


analytic, numerical_upwind = upwind_scheme()
numerical_lax = Lax_Wendroff()
plt.plot(x_vector, analytic[-1])
plt.plot(x_vector, numerical_upwind[-1])
#plt.plot(x_vector, numerical_lax[-1])
plt.legend(["Analytic", "Upwind scheme"])#, "Lax-Wendroff"])
plt.show()
#"""