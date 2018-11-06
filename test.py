import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from numba import jit
import random
from mpl_toolkits.mplot3d import Axes3D


def diagonal2(n):
    grid = [0 for j in range(n)]
    for i in range(n):
        grid[i] = i
    return grid

def shuffle2(grid, n, k, block_size, choices):
    plusmin = [0, 1, 2]

    return grid

def rotate(l, n):
    return l[n:] + l[:n]

def check_blocks2(n, grid):
    block_size = int(np.sqrt(n))
    for j in range(0, n - block_size, block_size):
        k_values = [c for c in range(block_size)]
        random.shuffle(k_values)
        a = int(j / block_size)
        for c in range(block_size):
            if grid[k_values[c]+ j] >= j:
                found = False
                while not found:
                    to = a * block_size + k_values[c]
                    if (grid[to] > j + block_size) or (grid[to] == to):
                        found = True
                    else:
                        k_values = rotate(k_values, -1)
                temp = grid[to]
                grid[to] = grid[k_values[c] + j]
                grid[k_values[c] + j] = temp
                a += 1
    return grid



def orthogonal2(n):
    grid = diagonal2(n)
    grid = check_blocks2(n, grid)
    x = np.linspace(0, n-1, n)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, grid, color='k')
    ax.hlines([3.5, 7.5, 11.5], 0, 16)
    ax.vlines([3.5, 7.5, 11.5], 0, 16)
#     ax.show()

orthogonal2(16)