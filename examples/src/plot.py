#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

def plot_and_save(filename):
    arr = np.loadtxt(filename, delimiter=" ", dtype=float)
    fig, ax = plt.subplots()
    ax.set(xlabel='x', ylabel='y',title=f'{filename}')
    plt.imshow(arr, cmap='RdBu', extent=[0, 1, 0, 1], origin="lower")
    plt.colorbar()
    fig.savefig(filename + ".png")
    plt.show()

plot_and_save("benchmark1.csv")
