import numpy as np 
import scipy
import matplotlib.pyplot as plt


"""Code that calculates the central and forward difference method of differentiation.
Compares the two.

Additionally, recursively calculates higher order derivatives using central difference method"""

def f(x):
    """Function to differentiate"""
    return np.exp(-x**2)


# Define all the step sizes that incrementally increase by an order

h_array = np.array([1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10,
                    1e-9, 1e-8, 1e-7, 1e-6, 1e-5,
                    1e-4, 1e-3, 1e-2, 1e-1, 1e0])



def central_difference(func, x, h):
    """Central Difference Formula"""
    return (func(x + h/2) - func(x - h/2)) / h
    
def forward_difference(func, x, h):
    """Forward Difference Formula"""
    return (func(x + h) - func(x)) / h


"""Part A,B,C: Compare Central and Forward Difference"""

central_difference_array = np.zeros(len(h_array))  # Initialize array to store central difference results
forward_difference_array = np.zeros(len(h_array))  # Initialize array to store forward difference results

x = 0.5  # point at which to differentiate

"""Analytical value"""
analyticalValue = (4* 0.5**2 - 2) * np.exp(-0.5**2)   # Analytical derivative of f(x) = e^(-x^2) at x = 0.5
print(f"Analytical Derivative : {analyticalValue}")


print("Calculated Derivative")
for i in range(len(h_array)):
    h = h_array[i]

    # Compute the derivative for each h
    central_difference_array[i] = central_difference(f, x, h)
    forward_difference_array[i] = forward_difference(f, x, h)

    print(f"h = {h}: Central Difference = {central_difference_array[i]} \t Forward Difference = {forward_difference_array[i]}")


"""Finding the Relative error compared to the analytical value"""

central_relative_error = np.zeros(len(h_array))  # Initialize array for relative error of central difference
forward_relative_error = np.zeros(len(h_array))  # Initialize array for relative error of forward difference

central_relative_error = (central_difference_array - analyticalValue)/analyticalValue
forward_relative_error = (forward_difference_array - analyticalValue)/analyticalValue

print("\n\nRelative Error")
for i in range(len(h_array)):
    h = h_array[i]

    print(f"h = {h}: Central Difference = {central_relative_error[i]} \t Forward Difference = {forward_relative_error[i]}")





"""Part D : Plot the relative error in log plot"""

plt.plot(h_array, np.abs(central_relative_error), label='Central')
plt.plot(h_array, np.abs(forward_relative_error), label='Forward')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('h')
plt.legend()
plt.savefig("Lab2Q3D.png")




"""Part F: Investigate Higher Derivative"""
def g(x):
    """Function to test higher derivative"""
    return np.exp(2*x)


def higher_derivative(func, x, m, h):
    """Recursive calculation for higher-order derivatives, given in the question sheet"""
    if m > 1:
        return (higher_derivative(func, x + h/2, m - 1, h) - higher_derivative(func, x - h/2, m-1, h))/(h)
    else:
        return central_difference(func, x, h)  # Base case: use central difference for m=1


h= 1e-6  # step size
x = 0  # Point to calculate the derivative

print("\n\nFirst 5 derivatives")
for i in range(1, 6):
    print(f"The derivative of order {i} is {higher_derivative(g, x, i, h)}, expected value is {2.**(i)}")  # check with real value

