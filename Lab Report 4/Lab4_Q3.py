import numpy as np
import matplotlib.pyplot as plt

# Initial Parameters
epsilon = 1
deltax = 0.02  # [m]
deltat = 0.005  # [s]
Lx = 2 * np.pi  # [m]

Tf = 2  # [s]

# Initial arrays (t = 0)
x = np.arange(0, Lx, deltax)
N = len(x) - 1
u = np.sin(x)

# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot of u vs x at time = 0
plt.plot(x, u, label="t=0s")

"""Implement FTCS method"""

beta = epsilon * deltat / deltax

# Time loop
epsilon = deltat / 100
t1 = 0.5  # [s]
t2 = 1  # [s]
t3 = 1.5  # [s]
tend = Tf + epsilon

t = 0

# define two arrays for u
u1 = u
u2 = u - deltat * u * np.cos(x)

while t < tend:
    t = t + deltat

    # Boundary condition satisfy
    u1[0], u2[0], u1[-1], u2[-1] = 0, 0, 0, 0

    u_new = np.zeros(len(x))
    u2_squared = u2**2

    u_new[1:N] = u1[1:N] - beta / 2 * (u2_squared[2 : N + 1] - u2_squared[0 : N - 1])

    u1, u2 = u2.copy(), u_new.copy()

    # Plot of u vs x at specified times
    if abs(t - t1) < epsilon:
        print(f"Max u2 at t={t1}: x = {x[np.argmax(u2)]}")
        plt.plot(x, u2, label=f"t={t1}s")

    if abs(t - t2) < epsilon:
        print(f"Max u2 at t={t2}: x = {x[np.argmax(u2)]}")
        plt.plot(x, u2, label=f"t={t2}s")

    if abs(t - t3) < epsilon:
        print(f"Max u2 at t={t3}: x = {x[np.argmax(u2)]}")
        plt.plot(x, u2, label=f"t={t3}s")

# Finalize the plot
plt.xlabel("x [m]")
plt.ylabel("u [m/s]")
plt.title("u vs x at Different Times")
plt.legend()
plt.savefig("Lab4Q3_combined.png")
plt.show()
