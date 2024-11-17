import numpy as np
import matplotlib.pyplot as plt
import time

"""Code for Q1: Plots the Potential Map and Electric Field
associated with a parallel plate capacitor in a grounded box.
Uses relaxation and overrelaxation methods coupled with Gauss-Seidel."""

plt.rcParams.update(
    {"figure.figsize": [12, 8], "font.size": 20}
)  # Large figure and font sizes


def LaplacianSolver(initial_phi, target, w=0):
    """
    Solves the Laplace equation using the relaxation/overrelaxation method and Gauss Seidel.

    a non-zero w parameter will use the overrelaxation method
    """

    # Return array
    phi = initial_phi.copy()
    M = len(phi)

    # Main loop
    delta = 1.0

    while delta > target:

        old_values = phi.copy()
        # Calculate new values of the potential
        for i in range(M - 1):
            for j in range(M - 1):
                if i == 0 or i == M or j == 0 or j == M or j == 20 or j == 80:
                    continue  # phi is the same
                else:
                    # main equation
                    if w:  # overrrlaxation
                        phi[i, j] = (
                            phi[i + 1, j]
                            + phi[i - 1, j]
                            + phi[i, j + 1]
                            + phi[i, j - 1]
                        ) * (1 + w) / 4 - w * phi[i, j]
                    else:  # relaxation
                        phi[i, j] = (
                            phi[i + 1, j]
                            + phi[i - 1, j]
                            + phi[i, j + 1]
                            + phi[i, j - 1]
                        ) / 4

        # Calculate maximum difference from old values
        delta = np.max(
            abs(phi - old_values)
        )  # because we are checking if both values are equal (relaxation)

    return phi


# Constants
M = 100  # Grid squares on a side
target = 1e-6  # Target accuracy


# Create arrays to hold potential values
initial_phi = np.zeros([M, M], float)
initial_phi[:, 20] = 1  # 2cm from left is 1V
initial_phi[:, -20] = -1  # 2cm from right is -1V


"""Part A: Relaxation Method"""
# Calculate the potential
start = time.time()
phi_relaxed = LaplacianSolver(initial_phi, target)
print("Time Taken for w = 0 is ", time.time() - start)

plt.contourf(phi_relaxed, cmap="RdBu")
plt.colorbar()
plt.title("Potential Map of Parallel Plate Capacitor (relaxation method)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
# plt.show()
plt.savefig("Lab4Q1a.png")
plt.clf()


"""Part B: Overrelaxation Method"""

"""w = 0.1"""

# Calculate the potential
start = time.time()
phi_overrelaxed1 = LaplacianSolver(initial_phi, target, w=0.1)
print("Time Taken for w = 0.1 is ", time.time() - start)

plt.contourf(phi_overrelaxed1, cmap="RdBu")
plt.colorbar()
plt.title("Potential Map of Parallel Plate Capacitor (w=0.1)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
# plt.show()
plt.savefig("Lab4Q1b1.png")
plt.clf()


"""w=0.5"""

# Calculate the potential
start = time.time()
phi_overrelaxed2 = LaplacianSolver(initial_phi, target, w=0.5)
print("Time Taken for w = 0.5 is ", time.time() - start)

plt.contourf(phi_overrelaxed2, cmap="RdBu")
plt.colorbar()
plt.title("Potential Map of Parallel Plate Capacitor (w=0.5)")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
# plt.show()
plt.savefig("Lab4Q1b2.png")
plt.clf()


""" Part C: StreamLines"""
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)

Ey, Ex = np.gradient(-phi_relaxed, y, x)  # E = -grad(phi)

strm = plt.streamplot(X, Y, Ex, Ey, color=phi_relaxed, linewidth=2, cmap="RdBu")
cbar = plt.colorbar(strm.lines)
cbar.set_label("Potential $V$")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title("Streamlines of Electric Field")
plt.axis("equal")
plt.tight_layout()
# plt.show()
plt.savefig("Lab4Q1c.png")
