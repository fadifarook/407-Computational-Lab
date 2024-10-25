import numpy as np
import scipy
import matplotlib.pyplot as plt
from gaussxw import gaussxw, gaussianQuadrature


"""Code that calculates the time period of a relativistic particle on a spring.
Also compares this to classical and relativistic limit.

Imports the function gaussianQuadrature from gaussxw module I wrote. It calculates the integral and returns
xpoints, weightedpoints and integral in a tuple of that order """


"""Part A"""


def g(x, k, m, x0):
    """Function for velocity, v = dx/dt"""

    c = 3e8  # speed of light

    temp_val = (2 * m * c**2) + (k * (x0**2 - x**2) / 2)
    temp_val *= k * (x0**2 - x**2)

    temp_val /= 2 * ((m * c**2) + (k * (x0**2 - x**2) / 2)) ** 2

    return c * temp_val**0.5


def T(x, k, m, x0):
    """Integrand for time period calculation"""
    return 4 / g(x, k, m, x0)  # The integrand is 4 times the reciprocal of velocity


m = 1  # Mass of the particle [kg]
k = 12  # Spring constant [N/m]

x0 = 0.01  # Maximum displacement [m]

# Number of sample points
N1 = 8
N2 = 16

# Generate the x and weight arrays for Gaussian quadrature once
x8_initial, w8_initial = gaussxw(N1)
x16_initial, w16_initial = gaussxw(N2)

# Perform Gaussian quadrature to calculate the time period integral for N1 and N2 points
xp8, wp8, s8 = gaussianQuadrature(T, N1, 0, x0, x8_initial, w8_initial, k, m, x0)
xp16, wp16, s16 = gaussianQuadrature(T, N2, 0, x0, x16_initial, w16_initial, k, m, x0)

print("\n\nPart A: Calculate Time Period")
print(f"Time period integral with \t N = 8 : {s8} \t N = 16 {s16}")
print(f"Estimated Fractional Error: = {(s16-s8)/s16}\n")


"""Part B: Plot integrand and weighted integrand"""

x_array = np.linspace(0, x0, 100)  # Generate near-continous points between 0 and x0

# Plot the time period integrand T(x) as a function of x
plt.plot(x_array, T(x_array, k, m, x0))

plt.legend()
plt.xlabel("x value [m]")
plt.ylabel("$\\frac{4}{g_k}$ Value [s/m]")
plt.title("Integrand Value over x")
plt.savefig("Lab2Q2B1")
plt.clf()

# Plot the Gaussian quadrature points and corresponding weighted values for N = 8 and N = 16
plt.scatter(xp8, wp8 * T(xp8, k, m, x0), label="N=8")
plt.scatter(xp16, wp16 * T(xp16, k, m, x0), label="N=16")

plt.legend()
plt.xlabel("x value [m]")
plt.ylabel("$\\frac{4w_k}{g_k}$ Value [s]")
plt.title("Weighted Values over x")
plt.savefig("Lab2Q2B2")
plt.clf()


"""Part C: T as a function of x0"""

c = 3e8  # Speed of light [m/s]
k = 12  # Spring constant [N/m]
m = 1  # Mass of particle [kg]
xc = c * np.sqrt(m / k)  # Relativistic amplitude

x0_array = np.linspace(1, 10 * xc, 50)  # Generate an array of x0 values from 1 to 10*xc

N = 16  # number of sample points

T_array = np.zeros(len(x0_array))  # Array to store time period results

# Calculate the time period for each value of x0 in the array
for i in range(len(x0_array)):
    T_array[i] = gaussianQuadrature(
        T, N, 0, x0_array[i], x16_initial, w16_initial, k, m, x0_array[i]
    )[2]
    # Reused x16 and w16 gaussian quadrature calculation from earlier

plt.plot(x0_array, T_array, label="Integral Calculation")
plt.axhline(
    2 * np.pi * np.sqrt(m / k), label="Classical Limit T = $2\pi \sqrt{m/k}$", color="k"
)  # Plots Classical Limit
plt.plot(
    x0_array, 4 * x0_array / c, label="Relativistic Limit T = 4$x_0$/c", color="r"
)  # Plots relativistic limit
plt.ylabel("Time Period [s]")
plt.xlabel("x0 value [m]")
plt.title("Dependence of Time period on Amplitude $x_0$")
plt.legend()
plt.savefig("Lab2Q2C")
