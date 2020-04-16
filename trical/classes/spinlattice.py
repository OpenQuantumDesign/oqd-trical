from .base import Base
from ..misc import constants as cst
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
        pass

    def plot_interaction_graph(self, **kwargs):
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

        params = {"dir": "x", "k": np.pi * 2 / 355e-9}
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

        a = {"x": 0, "y": 1, "z": 2}[self.dir]
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
