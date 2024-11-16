import numpy as np
import matplotlib.pyplot as plt

# Initial Parameters
L = 1  # Length [m]
g = 9.81  # Acceleration due to gravity [m/s^2]

deltax = 0.02  # [m]
deltat = 0.01  # [s]

x = np.arange(0, L, deltax)
J = len(x) - 1

# Flat Bottom Topography
nb = np.zeros(len(x))  # flat bottom
H = 0.01  # [m]  Free surface altitude at rest

# Parameters for eta definition
A = 0.002  # [m]
mean = 0.5  # [m]
sigma = 0.05  # [m]

# Main arrays (at time 0)
u = np.zeros(len(x))
helper_value = np.exp(-((x - mean) ** 2) / (sigma**2))
eta = H + A * helper_value - np.mean(A * helper_value)

# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot of eta vs x at time = 0
plt.plot(x, eta, label="t=0s")

"""Implement FTCS method"""

c = deltat / (2 * deltax)

# Time loop
epsilon = deltat / 100
t1 = 1  # [s]
t2 = 4  # [s]
tend = t2 + epsilon

t = 0

while t < tend:
    t = t + deltat

    u[0], u[-1] = 0, 0
    # eta[0], eta[-1] = H, H

    u_squared = u**2
    u_new, eta_new = np.zeros(len(x)), np.zeros(len(x))

    u_new[0] = u[0] - deltat / deltax * (
        0.5 * u_squared[1] + g * eta[1] - 0.5 * u_squared[0] - g * eta[0]
    )
    eta_new[0] = eta[0] - deltat / deltax * ((eta[1] * u[1]) - (eta[0] * u[0]))

    u_new[J] = u[J] - deltat / deltax * (
        0.5 * u_squared[J] + g * eta[J] - 0.5 * u_squared[J - 1] - g * eta[J - 1]
    )
    eta_new[J] = eta[J] - deltat / deltax * ((eta[J] * u[J]) - (eta[J - 1] * u[J - 1]))

    u_new[1:J] = u[1:J] - c * (
        0.5 * u_squared[2 : J + 1]
        + g * eta[2 : J + 1]
        - 0.5 * u_squared[0 : J - 1]
        - g * eta[0 : J - 1]
    )
    eta_new[1:J] = eta[1:J] - c * (
        (eta[2 : J + 1] * u[2 : J + 1]) - eta[0 : J - 1] * u[0 : J - 1]
    )

    u, eta = u_new.copy(), eta_new.copy()

    # Now plot at appropriate times
    if abs(t - t1) < epsilon:
        plt.plot(x, eta, label="t=1s")

    if abs(t - t2) < epsilon:
        plt.plot(x, eta, label="t=4s")

# Finalize the plot
plt.xlabel("x [m]")
plt.ylabel("eta [m]")
plt.title("Free Surface Height vs x at Different Times")
plt.legend()
plt.savefig("Lab4Q2_combined.png")
plt.show()
