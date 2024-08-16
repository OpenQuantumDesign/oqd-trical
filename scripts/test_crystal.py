import sys

import torch
from torch import optim

from rich import print as pprint

import matplotlib
from matplotlib import colors, cm, patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import seaborn as sns

########################################################################################

import trical

########################################################################################

matplotlib.rcParams["figure.figsize"] = (12, 8)
matplotlib.rcParams["font.size"] = 24
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

########################################################################################

k = 3
N = 100


def f(x):
    trap = (
        0.5 * (2 * torch.pi * torch.tensor([1, 1, 2]).to(x))[:, None] ** 2 * x**2 * 2
    ).sum()

    return trap


p = trical.Potential(f) + trical.CoulombPotential()
crys = trical.IonCrystal(p, N=N, dim=k)

########################################################################################

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(*crys.equilibrium_position.detach(), "o")
ax.set_aspect("equal")

plt.show()
