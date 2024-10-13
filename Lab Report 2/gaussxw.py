from pylab import *


"""Code that defines functions used in Gaussian Quadrature integration calculation

Contains gaussxw, and gaussxwab that provides x values and weights, that is lifted from
lab exercises and textbook

Also contins gaussianQuadrature, a function that I wrote to calculate the integral using
the other two functions.
"""

def gaussxw(N):

    """Gaussian Quadrature helper function that returns the weights and
    corresponding x values between 0 and 1
    
    Provided function from class and textbook"""

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

    return x,w  # returns the roots and weights

def gaussxwab(a,b, x_initial, w_initial):
    """Converts the values gained from gaussxw function (0 to 1) 
    to different limits a to b"""

    return 0.5*(b-a)*x_initial+0.5*(b+a),0.5*(b-a)*w_initial  # Transform the interval from [0,1] to [a,b]



def gaussianQuadrature(func, N, a, b, x_initial, w_initial, *params):
    """Calculates the gaussian weights, x values and the integral when given
    the function, number of points N and two limits a,b
    
    *params is a placeholder for all the other parameters (in order) that is taken by the function"""

    xp, wp = gaussxwab(a, b, x_initial, w_initial)  # Retrieves the x values and weights from helper functions

    s = 0.0  # Initialize integral value

    for  i in range(N):
        s += wp[i] * func(xp[i], *params)   # equation for guassian quadrature calculation

    return xp, wp, s  # Returns the x values, weights and computer integral value in a tuple