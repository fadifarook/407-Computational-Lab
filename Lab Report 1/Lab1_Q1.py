import numpy as np
import matplotlib.pyplot as plt
import os


"""This question involved a comparison of errors when using two equations of finding standard
deviation

It takes data from cdata.txt or creates an array of normal values, and returns the standard deviation
using multiple methods.
"""


print("Part B")


def std_1(array):
    """Function corresponding to Standard Deviation Equation 1 (Two pass)"""

    mean = np.mean(array)
    temporary_value = np.sum((array - mean) ** 2)  # intermediate step

    return np.sqrt(temporary_value / (len(array) - 1))


def std_2(array):
    """Function corresponding to Standard Deviation Equation 2 (One pass)"""

    mean = np.mean(array)

    temporary_value = np.sum(array**2)
    temporary_value -= len(array) * mean**2

    if temporary_value < 0:
        print("Negative Square Root")  # Negative check
        temporary_value *= -1

    return np.sqrt(temporary_value / (len(array) - 1))


# Load the data from the file
speed = np.loadtxt("cdata.txt")

true_value = np.std(speed, ddof=1)
std1_value = std_1(speed)
std2_value = std_2(speed)

print("Numpy Standard Deviation: ", true_value)
print(
    "Equation 1 (two pass) Standard Deviation: ",
    std1_value,
    "\t Relative Error: ",
    (true_value - std1_value) / true_value,
)
print(
    "Equation 2 (one pass) Standard Deviation: ",
    std2_value,
    "\t Relative Error: ",
    (true_value - std2_value) / true_value,
)


print("\n\nPart C")

# Create an array with normal distribuion with same sigma and number of values (n)
# But different mean1 and mean2
mean1 = 0.0
mean2 = 1.0e7
sigma = 1.0
n = 2000

# Create arrays
normalSequence1 = np.random.normal(mean1, sigma, 2000)
normalSequence2 = np.random.normal(mean2, sigma, 2000)


true_value1 = np.std(normalSequence1, ddof=1)
std1_value_mean1 = std_1(normalSequence1)
std2_value_mean1 = std_2(normalSequence1)

print("Mean = 0")
print("True Standard Deviation", true_value1)
print(
    "Equation 1 (Two Pass) Standard Deviation: ",
    std1_value_mean1,
    "\t Relative Error",
    (std1_value_mean1 - true_value1) / true_value1,
)
print(
    "Equation 2 (One Pass) Standard Deviation: ",
    std2_value_mean1,
    "\t Relative Error",
    (std2_value_mean1 - true_value1) / true_value1,
)

print("\nMean = 1e7")
true_value2 = np.std(normalSequence2, ddof=1)
std1_value_mean2 = std_1(normalSequence2)
std2_value_mean2 = std_2(normalSequence2)
print("True Standard Deviation", true_value2)
print(
    "Equation 1 (Two Pass) Standard Deviation: ",
    std1_value_mean2,
    "\t Relative Error",
    (std1_value_mean2 - true_value2) / true_value2,
)
print(
    "Equation 2 (One Pass) Standard Deviation: ",
    std2_value_mean2,
    "\t Relative Error",
    (std2_value_mean2 - true_value2) / true_value2,
)


print("\n\nPart D")


def std_3(array):
    """New standard deviation calculation where we subtract a random value"""

    # Subtract a random value beforehand
    randomValue = np.random.choice(array)
    array -= randomValue

    mean = np.mean(array)

    temporary_value = np.sum(array**2)
    temporary_value -= len(array) * mean**2

    if temporary_value < 0:
        print("Negative Square Root")  # Negative check
        temporary_value *= -1

    return np.sqrt(temporary_value / (len(array) - 1))


print("With modified standard deviation calculation:")
std3_value_mean1 = std_3(normalSequence1)
std3_value_mean2 = std_3(normalSequence2)
print(
    "Mean = 0 ; New Standard Deviation: ",
    std3_value_mean1,
    "\t Relative Error: ",
    (std3_value_mean1 - true_value1) / true_value1,
)
print(
    "Mean = 1e7 ; New Standard Deviation: ",
    std3_value_mean2,
    "\t Relative Error: ",
    (std3_value_mean2 - true_value2) / true_value2,
)
