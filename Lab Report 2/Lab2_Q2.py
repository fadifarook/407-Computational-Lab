import numpy as np 
import scipy
import matplotlib.pyplot as plt


"""Realtivistic Spring Particle"""

# %load gaussxw
from pylab import *
def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15  # machine precision is 1e-16
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w




"""Part A"""

def g(x, k, m, x0):
    """Function for v = dx/dt"""

    c = 3e8

    temp_val = (2*m*c**2) + (k * (x0**2 - x**2) / 2)
    temp_val *= k * (x0**2 - x**2)

    temp_val /= 2 * ((m*c**2) + (k * (x0**2 - x**2) / 2))**2

    return c * temp_val**0.5


def T(x, k, m, x0):
    return 4 / g(x, k, m, x0)

m = 1  #[kg]
k = 12  #[N/m]

x0 = 0.01  # [m]


N1 = 8
N2 = 16

xp1, wp1 = gaussxwab(N1, 0, x0)
xp2, wp2 = gaussxwab(N2, 0, x0)

s1 = 0.0
for  i in range(N1):
    s1 += wp1[i] * T(xp1[i], k, m, x0)

s2 = 0.0
for  i in range(N2):
    s2 += wp2[i] * T(xp2[i], k, m, x0)

print(f"N = 8 : {s1} \t N = 16 {s2}")





"""Part B: plot T"""

x_array = np.linspace(0, 0.01, 100)

plt.plot(x_array, T(x_array, k, m, x0), label = 'T')
plt.scatter(xp1, T(xp1, k, m, x0), label='8')
plt.scatter(xp2, T(xp2, k, m, x0), label='16')

plt.legend()
plt.show()