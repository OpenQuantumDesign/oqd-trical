import os

from rich import print as pprint

import torch
from torch.optim import Adam

import matplotlib
from matplotlib import colors, cm, patches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import seaborn as sns

########################################################################################

matplotlib.rcParams["figure.figsize"] = (12, 8)
matplotlib.rcParams["font.size"] = 24
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"

########################################################################################

N = 10
k = 3
x = torch.rand((k * N)).to("cuda")
x.requires_grad_(True)

########################################################################################


def f(x):
    x = x.reshape(k, N)

    trap = (
        0.5 * (2 * torch.pi * torch.tensor([1, 1, 2]).to(x))[:, None] ** 2 * x**2 * 2
    ).sum()

    idcs = torch.triu_indices(N, N, 1)
    x1 = x[:, idcs[0]]
    x2 = x[:, idcs[1]]
    coulomb = (2 / torch.norm(x1 - x2, dim=0)).sum(0)

    return trap + coulomb


########################################################################################


# G = torch.autograd.grad(f(x), x)
# H = torch.autograd.functional.hessian(f, x)

# pprint(G)
# pprint(H)

########################################################################################

x = torch.nn.Parameter(torch.empty(k * N))
torch.nn.init.normal_(x)

max_epochs = 1000
lr = 1e-2
optimizer = Adam(
    [
        x,
    ],
    lr=lr,
)

for i in range(max_epochs):
    optimizer.zero_grad()

    loss = f(x)
    loss.backward()

    optimizer.step()

    print(f"\rEpoch {i}", end="")

########################################################################################

x = x.reshape(k, N).detach()

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(*x, "o")
ax.set_aspect("equal")

plt.show()
