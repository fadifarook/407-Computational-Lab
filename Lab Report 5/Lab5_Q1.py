import numpy as np
import matplotlib.pyplot as plt

"""Code for Q1: Brownian Motion and Diffusion Limited Aggregation"""


""" Part A"""

# Initial Parameters
L = 101  # Grid size
N = 5000  # Number of time steps

grid = np.zeros((L, L))
x_array = np.linspace(-L // 2, L // 2, L)
y_array = np.linspace(-L // 2, L // 2, L)


def choose_direction():
    """Choose a random direction up [0,1], down [0,-1], right [1,0], left [-1,0]"""
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    return directions[np.random.choice(len(directions))]


# Simulate Brownian Motion

x_pos = []
y_pos = []

x, y = L // 2, L // 2
for _ in range(N):
    dx, dy = choose_direction()
    # If max or min index is reached, choose a new direction
    while (x + dx > L - 1) or (x + dx < 0) or (y + dy > L - 1) or (y + dy < 0):
        dx, dy = choose_direction()

    x += dx
    y += dy

    x_pos.append(x)
    y_pos.append(y)

    grid[x, y] = 1

# plot grid
plt.contourf(x_array, y_array, grid, cmap="hot")
plt.colorbar()
# plt.show()
plt.savefig("Plots/Lab5Q1a1.png")
plt.clf()

# Plot Brownian Motion
time_array = np.linspace(0, N, N) / 1000
x_pos = np.array(x_pos) - L // 2
y_pos = np.array(y_pos) - L // 2

plt.plot(time_array, x_pos, label="x")
plt.plot(time_array, y_pos, label="y")
plt.xlabel("Time (s)")
plt.ylabel("Position")
plt.legend()
# plt.show()
plt.savefig("Plots/Lab5Q1a2.png")
plt.clf()


""" Part B """


def check_neighbors(x, y, grid):
    """Check if any neighbors are occupied, or if the cell is outside grid
    Return True if there is a neighbor or at edge, False otherwise"""

    if x == 0 or x == L - 1 or y == 0 or y == L - 1:
        return True

    if (
        grid[x - 1, y] == 1
        or grid[x + 1, y] == 1
        or grid[x, y - 1] == 1
        or grid[x, y + 1] == 1
    ):
        return True

    return False


# Simulate Diffusion Limited Aggregation

grid = np.zeros((L, L))
x, y = L // 2, L // 2  # start at center

while grid[L // 2, L // 2] == 0:
    # Move x + dy or y+dy until check_neighbors returns True
    while not check_neighbors(x, y, grid):
        dx, dy = choose_direction()
        x += dx
        y += dy

    grid[x, y] = 1
    x, y = L // 2, L // 2  # reset to center


# plot grid
# contourf with binary cmap
plt.contourf(x_array, y_array, grid, cmap="Greys")
plt.colorbar()
# plt.show()
plt.savefig("Plots/Lab5Q1b1.png")
plt.clf()
