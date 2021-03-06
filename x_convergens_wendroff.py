import numpy as np
import matplotlib.pyplot as plt
import readwrite as rw

"""
h = 0.0037
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
N = 1000
"""
# Lager konvergensplot for Lax-Wendroff i x-retning

#h_values = [2**i for i in range(4, 8)]
h_values = [2**i for i in range(3,9)]
#h_values = [10, 20, 50, 100, 200, 500]
number_of_discretizations = len(h_values)
errors1 = np.zeros(number_of_discretizations) # Error in 1-norm
errors2 = np.zeros(number_of_discretizations) # Error in 2-norm
ref_sol = rw.read_data("u_lax_wendroff_x2.txt") # Reference solution
L = 2**13
ref_sol_points = L/(len(ref_sol[0])-1)
k = 10**-4

sigma = 300
tau = 0.5
V_0 = 2000
rho_hat = 0.14
E = 100
c_0 = 900
mu = 10000
f_up = 32.5
f_rmp = 2
rho_up = 0.02
N = 10**4

def q(t):
    #if (t>0.01):
        #return 0
    return f_rmp

def phi(x):
    return ((2*np.pi*(sigma**2))**(-1/2))*np.exp(-(x**2)/(2*(sigma**2)))

def V_ro(ro):
    return V_0*(1-(ro/rho_hat))/(1+E*((ro/rho_hat)**4))

def s(U,m,n,h):
    u1 = q(n*k)*phi((m*h)-(L/2))
    u2 = ((V_ro(U[0,m])-U[1,m])/tau)
    return np.array([u1, u2])

def f_u(U,m):
    u1 = U[0,m]*U[1,m]
    u2 = (1/2)*(U[1,m]**2) + (c_0**2)*np.log(U[0,m])
    return np.array([u1,u2])

def x_convergence(h):
    x = np.linspace(-L / 2, L / 2, int(L / h) + 1)
    u = np.zeros([2,len(x)+1])
    u_next = np.zeros([2,len(x)+1])
    u_half = np.zeros([2,len(x)+1])
    initial_velocity = V_ro(rho_up)
    u_half[0,:] = rho_up
    u_half[1,:] = initial_velocity
    u[0,:] = rho_up
    u[1,:] = initial_velocity
    u_next[0,:] = rho_up
    u_next[1,:] = initial_velocity
    for n in range(N):
        if n % 100 == 0:
            print(int(100 * n / N), "%")

        for m in range(1,len(x)-1):
            s_next = s(u,m,n,h)
            s_next_p1 = s(u, m+1, n, h)
            f_next = f_u(u,m) #m-1
            f_next_p1 = f_u(u,m+1) #m+1
            u_half[0,m] = ((u[0,m]+ u[0,m+1])/2) -(k/(2*h))*(f_next_p1[0]-f_next[0])+((k/2)*(s_next[0]+s_next_p1[0]))
            u_half[1,m] = ((u[1,m]+ u[1,m+1])/2) -(k/(2*h))*(f_next_p1[1]-f_next[1])+((k/2)*(s_next[1]+s_next_p1[1]))
            f_half_p1 = f_u(u_half,m)
            f_half_m1 = f_u(u_half,m-1)
            s_half_p1 = s(u_half, m, n,h)
            s_half_m1 = s(u_half, m-1, n,h)
            u_next[0,m]= u[0,m] -(k/(2*h))*(f_half_p1[0]-f_half_m1[0])+((k/2)*(s_half_p1[0]+s_half_m1[0]))
            u_next[1,m] = u[1,m] -(k/(2*h))*(f_half_p1[1]-f_half_m1[1])+((k/2)*(s_half_p1[1]+s_half_m1[1]))


        u_next[0,0] = u_next[0,1] -(u_next[0,2]-u_next[0,1])
        u_next[1,0] = u_next[1,1] - (u_next[1,2]-u_next[1,1])

        u_next[0, len(x)] = u_next[0, len(x)-1]+(u_next[0,len(x)-1]-u_next[0, len(x)-2])
        u_next[1, len(x)] = u_next[1, len(x)-1]+(u_next[1,len(x)-1]-u_next[1, len(x)-2])
        u = u_next
        #u[:, len(x)] = u[:, len(x) - 1]
        #u[:, 0] = u[:, 1]
    plt.plot(x, u[0,:-1])
    plt.show()
    error = np.zeros(len(ref_sol[0]))
    print("length of reference:", len(ref_sol[0]))
    print("lenght of u:", len(u[0]))
    #relative_discretization = int((len(ref_sol[0]) - 1) / (len(u[0]) - 2))
    #print("relationship:", ((len(ref_sol[0]) - 1) / (len(u[0]) - 2)))
    for i in range(1, len(ref_sol[0])-1):
        error[i] = np.interp(ref_sol_points*i-L/2, x, u[0,:-1]) - ref_sol[0][i]
        # print("reference:", ref_sol[0][i*relative_discretization])
    return h*np.linalg.norm(ord=1, x=error), np.sqrt(h)*np.linalg.norm(ord = 2, x = error)


if __name__ == "__main__":
    for i in range(number_of_discretizations):
        print("h:", h_values[i])
        errors1[i],errors2[i] = x_convergence(h_values[i])
        print("feil i 1-norm:",errors1[i])
        print("feil i 2-norm:",errors2[i])
        if i != 0:
            print("slope in 1-norm:", (errors1[i]/errors1[i-1])*h_values[i-1]/h_values[i])
            print("slope in 2-norm:", (errors2[i]/errors2[i-1])*h_values[i-1]/h_values[i], "\n")
    print("slope between first and last point in 1-norm:", (errors1[-1] / errors1[0]) * h_values[0] / h_values[-1])
    print("slope between first and last point in 2-norm:", (errors2[-1] / errors2[0]) * h_values[0] / h_values[-1])
    h_values = np.array(h_values)
    plt.loglog(h_values, errors1)
    #plt.loglog(h_values, errors2, "r")
    #plt.loglog(h_values, [h_values[i] for i in range,'r--')
    plt.loglog(h_values, errors1[0]/(h_values[0])*h_values, "r--")
    #plt.loglog(h_values, errors1[0]/((h_values[0]**2))*np.square(h_values), "b--")

    #plt.loglog(h_values, errors2[0]/((h_values[0]**2))*np.square(h_values), "r--")
    plt.xlabel("h")
    plt.legend(["Error in 1-norm","1st order reference"])
    plt.ylabel("Error")

    plt.show()
