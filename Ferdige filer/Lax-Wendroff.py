import numpy as np
import matplotlib.pyplot as plt
import readwrite as rw

#CONSTANTS
#The values are given in meters and minutes as units. Some are based on the numerical values used in the article
#"Dynamic states of a continuum traffic equation with on-ramp" , and some are adjusted to make the output graph better.
h = 5 #meters
k = 0.001 #minutes
L = 10000 #meters
x = np.linspace(-L/2,L/2,int(L/h)+1)
sigma = 300 #meters
tau = 0.5 #minutes
V_0 = 2000 #meters/minute
rho_hat = 0.14 #vehicles/meter
E = 100
c_0 = 900 #meter/minute
mu = 1000 #vehicles meter/minute
f_up = 32.5 #vehicles/minute
f_rmp = 2 #vehicles/minute
rho_up = 0.02 #vehicles/meter
N = 1000

#FUNCTIONS

def q(t): #total incoming flux
    #if (t>1):
        #return 0
    return f_rmp

def phi(x): #spatial distribution of the external flux
    return ((2*np.pi*(sigma**2))**(-1/2))*np.exp(-(x**2)/(2*(sigma**2)))

def V_ro(ro): #safe velocity
    return V_0*(1-(ro/rho_hat))/(1+E*((ro/rho_hat)**4))

def s(U,m,n): #source term
    u1 = q(n*k)*phi((m*h)-(L/2))
    u2 = ((V_ro(U[0,m])-U[1,m])/tau) + ((k/(h**2))*(U[1,m+1]-2*U[1,m]+U[1,m-1]))*(mu/U[0,m])
    #Last term central differences
    return np.array([u1, u2])

def f_u(U,m):
    u1 = U[0,m]*U[1,m]
    u2 = 1/2*(U[1,m]**2) + (c_0**2)*np.log(U[0,m])
    return np.array([u1,u2])

#DECLARATION OF INITIALIZATION OF U-VECTORS
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
for n in range(N): #Iterating over time-steps

    for m in range(1,len(x)): #Iterating over space-steps

        #EXECUTION OF LAX-WENDROFF SCHEME

        s_next = s(u,m,n)
        if m==(len(x)-1): #To avoid going out of range
            s_next_p1 = s(u, m, n)
        else:
            s_next_p1 = s(u, m+1, n)
        f_next = f_u(u,m)
        f_next_p1 = f_u(u,m+1)
        #Half step
        u_half[0,m] = ((u[0,m]+ u[0,m+1])/2) -(k/(2*h))*(f_next_p1[0]-f_next[0])+((k/2)*(s_next[0]+s_next_p1[0]))
        u_half[1,m] = ((u[1,m]+ u[1,m+1])/2) -(k/(2*h))*(f_next_p1[1]-f_next[1])+((k/2)*(s_next[1]+s_next_p1[1]))
        f_half_p1 = f_u(u_half,m)
        f_half_m1 = f_u(u_half,m-1)
        s_half_p1 = s(u_half, m, n)
        s_half_m1 = s(u_half, m-1, n)
        #Whole step
        u_next[0,m]= u[0,m] -(k/(2*h))*(f_half_p1[0]-f_half_m1[0])+((k/2)*(s_half_p1[0]+s_half_m1[0]))
        u_next[1,m] = u[1,m] -(k/(2*h))*(f_half_p1[1]-f_half_m1[1])+((k/2)*(s_half_p1[1]+s_half_m1[1]))

    #DEFINING BOUNDARY VALUES
    #Extrapolating from interior points
    #Left boundary
    u_next[0,0] = u_next[0,1] -(u_next[0,2]-u_next[0,1])
    u_next[1,0] = u_next[1,1] - (u_next[1,2]-u_next[1,1])
    #Right boundary
    u_next[0, len(x)] = u_next[0, len(x)-1]+(u_next[0,len(x)-1]-u_next[0, len(x)-2])
    u_next[1, len(x)] = u_next[1, len(x) - 1] + (u_next[1, len(x) - 1] - u_next[1, len(x) - 2])
    u = u_next

#SCALING VARIABLES
x = x/1000
u[0] = u[0]*1000
u[1] = u[1]*0.06

#PLOTTING
plt.plot(x,u[1][:-1])
plt.ylabel("Speed of vehicles, km/h")
plt.xlabel("x")
plt.show()

plt.figure()
plt.plot(x,u[0][:-1])
plt.ylabel("Density of vehicles, vehicles/km")
plt.xlabel("x")
plt.show()