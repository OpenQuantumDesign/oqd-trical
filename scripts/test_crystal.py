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
from trical import optimizer as optim

########################################################################################

matplotlib.rcParams["figure.figsize"] = (12, 8)
matplotlib.rcParams["font.size"] = 24
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

########################################################################################

k = 3
N = 250

p = trical.HarmonicPotential() + trical.CoulombPotential()
crys = trical.IonCrystal(p, N=N, dim=k)

pprint(crys.normal_modes[1][:, :, -1])

########################################################################################

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(*crys.equilibrium_position.detach(), "o")
ax.set_aspect("equal")

plt.show()
