from .base import Base
from ..misc import constants as cst
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class SpinLattice(Base):
    """
    Object representing a spin lattice system.

    :param J: Interaction graph of the spin lattice system.
    :type J: :obj:`numpy.ndarray`
    """

    def __init__(self, J):
        super(SpinLattice, self).__init__()

        self.J = J
        self.N = J.shape[0]
        pass

    def plot_interaction_graph(self, **kwargs):
        """Plots the normal mode frequencies of the ions

        Keyword Args:
            fig (matplotlib.figure.Figure): Figure for plot (default = plt.figure())
            idx (int): 3 digit integer representing position of the subplot
                (default = 111)
            plot_type(str): Type of plot (default = "bar3d")

        Returns:
            matplotlib.axes._subplots.Axes3DSubplot or
            matplotlib.axes._subplots.AxesSubplot: Axes of the plot
        """
        plot3d_params = {
            "fig": plt.figure() if "fig" not in kwargs.keys() else None,
            "idx": 111,
            "plot_type": "bar3d",
        }
        plot3d_params.update(kwargs)

        N = self.N
        Z = self.J

        if plot3d_params["plot_type"] == "bar3d":
            ax = plot3d_params["fig"].add_subplot(plot3d_params["idx"], projection="3d")

            Z = np.transpose(Z)

            X, Y = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, N - 1, N))

            X = X.flatten()
            Y = Y.flatten()
            Z = Z.flatten()

            W = Z - Z.min()
            frac = W / W.max()
            norm = colors.Normalize(frac.min(), frac.max())
            color = cm.gist_rainbow(norm(frac))

            ax.bar3d(X, Y, np.zeros(len(Z)), 1, 1, Z, color=color)
            ax.set_xlabel(r"$i$")
            ax.set_ylabel(r"$j$")
            ax.set_zlabel(r"$J$")
            ax.set_xticks(np.linspace(0.5, N - 0.5, N))
            ax.set_xticklabels(range(N))
            ax.set_yticks(np.linspace(0.5, N - 0.5, N))
            ax.set_yticklabels(range(N))
            ax.set_xlim(0, N)
            ax.set_ylim(0, N)
            ax.set_zlim(min(0, 1.1 * Z.min()), 1.1 * Z.max())
        elif plot3d_params["plot_type"] == "imshow":
            ax = plot3d_params["fig"].add_subplot(plot3d_params["idx"])
            ax.imshow(Z, cmap=cm.gist_rainbow)
            ax.set_xlabel(r"$j$")
            ax.set_ylabel(r"$i$")

        cax = plt.cm.ScalarMappable(cmap=cm.gist_rainbow)
        cax.set_array(Z)
        cbar = plot3d_params["fig"].colorbar(cax, ax=ax)
        cbar.set_label(r"$J$")
        return ax

    pass

    pass


class SimulatedSpinLattice(SpinLattice):
    """
    Object representing a spin lattice system simulated by a trapped ion system

    :param ti: A trapped ion system.
    :type ti: :obj:`trical.classes.trappedions.TrappedIons`
    :param mu: Raman beatnote detunings.
    :type mu: :obj:`numpy.ndarray`
    :param Omega: Rabi frequencies.
    :type Omega: :obj:`numpy.ndarray`
    """

    def __init__(self, ti, mu, Omega, **kwargs):
        self.ti = ti
        self.mu = np.array(mu)
        self.Omega = np.array(Omega)

        self.m = ti.m
        self.N = ti.N

        params = {"direc": "x", "k": np.pi * 2 / 355e-9}
        params.update(kwargs)
        self.__dict__.update(params)
        self.params = params

        if (
            np.isin(
                np.array(["x_pa", "w_pa", "b_pa"]), np.array(self.__dict__.keys())
            ).sum()
            != 3
        ):
            self.ti.principle_axis()

        a = {"x": 0, "y": 1, "z": 2}[self.direc]
        self.w = self.ti.w_pa[a * self.N : (a + 1) * self.N]
        self.b = self.ti.b_pa[
            a * self.N : (a + 1) * self.N, a * self.N : (a + 1) * self.N
        ]

        super(SimulatedSpinLattice, self).__init__(self.interaction_graph())
        pass

    def interaction_graph(self):
        """
        Calculates the interaction graph of the spin lattice simulated by the trapped ion system.

        :param J: Interaction graph of the spin lattice simulated by the trapped ion system.
        :type J: :obj:`numpy.ndarray`
        """
        try:
            len(self.m)
            eta = np.einsum(
                "in,in->in",
                self.b,
                2 * self.k * np.sqrt(cst.hbar / (2 * np.outer(self.m, self.w))),
            )
        except:
            eta = np.einsum(
                "in,n->in",
                self.b,
                2 * self.k * np.sqrt(cst.hbar / (2 * self.m * self.w)),
            )

        self.eta = eta
        zeta = np.einsum("im,in->imn", self.Omega, eta)
        self.zeta = zeta
        J = np.einsum(
            "ij,imn,jmn,n,mn->ij",
            1 - np.identity(self.N),
            zeta,
            zeta,
            self.w,
            1 / np.subtract.outer(self.mu ** 2, self.w ** 2),
        )
        return J

    def plot_raman_beatnote_detunings(self, **kwargs):
        pass

    def plot_rabi_frequencies(self, **kwargs):
        pass

    pass
