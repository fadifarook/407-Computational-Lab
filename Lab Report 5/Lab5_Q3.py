import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

"""Code for Q3: Mean Value integration and Importance Sampling"""


""" Part A: Mean Value Integration"""


def mean_value_integration(xmin, xmax, f, N):

    k = 0
    for i in range(N):
        x = (xmax - xmin) * np.random.random()
        k += f(x)

    return k * (xmax - xmin) / N


def f(x):
    return np.power(x, -0.5) / (1 + np.exp(x))


xmin = 0
xmax = 1
N = 10000  # Number of sample points

repeat = 1000  # change to 1000 at the end

integral_array = np.zeros(repeat)
for i in range(repeat):
    integral_array[i] = mean_value_integration(xmin, xmax, f, N)

print("Intergral (avg): ", np.mean(integral_array))

# Plot histogram
plt.hist(integral_array, bins=100)
plt.xlabel("Integral Value")
plt.ylabel("Frequency")
plt.title("Histogram of Integral Values")
plt.savefig("Plots/Lab5Q3a.png")
plt.clf()


"""Part B: Importance Sampling"""


def w(x):
    """Weighting function"""
    return np.power(x, -0.5)


def sampling():
    return np.random.random() ** 2


def importance_sampling(f, N, weighting_function, sampling_function, weight_integral):
    k = 0
    for i in range(N):
        x = sampling_function()
        k += f(x) / weighting_function(x)

    return k / N * weight_integral


weight_integral = 2

integral_array = np.zeros(repeat)
for i in range(repeat):
    integral_array[i] = importance_sampling(f, N, w, sampling, weight_integral)

print("Intergral (avg): ", np.mean(integral_array))

# Plot histogram
plt.hist(integral_array, bins=100)
plt.xlabel("Integral Value")
plt.ylabel("Frequency")
plt.title("Histogram of Integral Values")
plt.savefig("Plots/Lab5Q3b.png")
plt.clf()


""" Part D: New function"""

xmin = 0
xmax = 10
weight_integral = 1


def f_complicated(x):
    return np.exp(-2 * np.abs(x - 5))


def w_complicated(x):
    return np.exp(-((x - 5) ** 2) / 2) / np.sqrt(np.pi * 2)


def sampling_complicated():
    return np.random.normal(loc=5, scale=1)


integral_array = np.zeros(repeat)
for i in range(repeat):
    integral_array[i] = importance_sampling(
        f_complicated, N, w_complicated, sampling_complicated, weight_integral
    )

print("Intergral (avg): ", np.mean(integral_array))

# Plot histogram
plt.hist(integral_array, bins=100)
plt.xlabel("Integral Value")
plt.ylabel("Frequency")
plt.title("Histogram of Integral Values")
plt.savefig("Plots/Lab5Q3c.png")
plt.clf()
