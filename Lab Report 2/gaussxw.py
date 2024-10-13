# %load gaussxw
from pylab import *


def gaussxw(N):

    """Gaussian Quadrature helper function that returns the weights and
    corresponding x values between 0 and 1
    
    Same function from class and textbook"""

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
    """Converts the values gained from gaussxw function (0 to 1) 
    to different limits a to b"""

    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w  # formula for conversion



def gaussianQuadrature(func, N, a, b, *params):
    """Calculates the gaussian weights, x values and the integral when given
    the function, number of points N and two limits a,b
    
    *params is a placeholder for all the other parameters (in order) that is taken by the function"""

    xp, wp = gaussxwab(N, a, b)  # retrieves the x values and weights from helper functions

    s = 0.0  # integral value

    for  i in range(N):
        s += wp[i] * func(xp[i], *params)   # equation for guassian quadrature calculation

    return xp, wp, s