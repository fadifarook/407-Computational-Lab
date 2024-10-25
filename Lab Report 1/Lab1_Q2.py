import numpy as np
import matplotlib.pyplot as plt


def p(u):
    "Define the function p(u)"
    return (1 - u) ** 8


def q(u):
    "Define the function q(u)"
    return (
        1
        - 8 * u
        + 28 * u**2
        - 56 * u**3
        + 70 * u**4
        - 56 * u**5
        + 28 * u**6
        - 8 * u**7
        + u**8
    )


def xterms(u):
    "Create an array of all the terms in the summation, for a given u value"
    return np.array(
        [
            (1 - u) ** 8,
            -1,
            8 * u,
            -28 * u**2,
            56 * u**3,
            -70 * u**4,
            56 * u**5,
            -28 * u**6,
            8 * u**7,
            -(u**8),
        ]
    )


print("Part A")

N = 500  # number of terms
u_array = np.linspace(0.98, 1.02, N)

# Plot of p and q for u near 1.
plt.plot(u_array, p(u_array), color="#43a2ca", label="p(u)")
plt.plot(u_array, q(u_array), color="#a8ddb5", label="q(u)")  # noisier
plt.xlabel("u value")
plt.ylabel("Function Value")
plt.title("Plot of p and q for u near 1")
plt.legend()
plt.savefig("Q2PartA")
plt.clf()
# plt.show()


print("\n\nPart B")
difference_array = p(u_array) - q(u_array)
plt.plot(u_array, difference_array, color="black")
plt.title("Plot of difference in p and q for u near 1")
plt.xlabel("u value")
plt.ylabel("Difference Value")
plt.savefig("Q2PartBDifference")
plt.clf()
# plt.show()


"""Plot histogram"""
plt.hist(difference_array, bins=50, color="black")  # chose an arbitrary number of bins
plt.title("Plot of histogram of p-u for u near 1")
plt.xlabel("Noise Value")
plt.ylabel("Number of points")
plt.savefig("Q2PartBHist")
plt.clf()
# plt.show()


"""Calculation of error from the equation"""
C = 1e-16  # fractional error
numTerms = 10  # explained in report why this is 10
terms = xterms(u=1)  # all terms calculated at u = 1

squaredMeanTerms = np.mean(terms**2)
numpySTD = np.std(difference_array)
equationSTD = C * np.sqrt(numTerms) * np.sqrt(squaredMeanTerms)

print("Numpy Standard Deviation = ", numpySTD)
print("Equation Standard Deviation = ", equationSTD)
print(
    "Percentage Difference: ",
    round(abs(numpySTD - equationSTD) / abs(numpySTD) * 100),
    "%",
)


print("\n\nPart C")

# Plotting the fractional error from 0.980 to 0.984
u_array = np.linspace(0.980, 0.984, 100)
plt.plot(u_array, abs(p(u_array) - q(u_array)) / abs(p(u_array)), color="k")
plt.xlabel("u value")
plt.ylabel("Fractional Error")
plt.title("Fractional error less than 1")
plt.savefig("Q2PartC")
plt.clf()
# plt.show()


# Calculation of fractional error from equation at specific u
uGuess = 0.982
nearTerms = xterms(u=uGuess)
squarednearTerms = np.mean(nearTerms**2)

fractional_error = (
    C * np.sqrt(squarednearTerms) / (np.sqrt(numTerms) * np.mean(nearTerms))
)

print(f"Fractional Error at {uGuess}: ", abs(fractional_error) * 100, "%")


print("\n\nPart D")


def func_multiply(u):
    "Function to represent f equation (multiplication)"
    return (u**8) / ((u**4) * (u**4))


N = 500  # number of terms
u_array = u_array = np.linspace(0.98, 1.02, N)

# Plot f-1 near u=1
plt.plot(u_array, func_multiply(u_array) - 1.0, color="k")
plt.xlabel("u value")
plt.ylabel("f-1 value")
plt.title("f-1 values near u=1")
plt.savefig("Q2PartD")
plt.clf()
# plt.show()


# Compare the equation and true value of standard deviation for multiplication
print("Numpy Standard Deviation: ", np.std(func_multiply(u_array) - 1.0))
print("Equation Standard Deviation: ", C * func_multiply(1))
