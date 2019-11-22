"""
Defines the SpinLattice class and its subclasses representing a spin lattice system or a
simulation of a spin lattice system
"""
from .. import constants as cst
import numpy as np


class SpinLattice(object):

    """
    Object representing a spin lattice system
    
    Attributes:
        J (2-D array of float): Interaction graph of the system
    """

    def __init__(self, J):
        """
        Initialization function for a SpinLattice object
        
        Args:
            J (2-D array of float): Interaction graph of the system
        """
        super(SpinLattice, self).__init__()

        self.J = J
        pass

    def plot_interaction_graph(self, **kwargs):
        pass

    pass


class SimulatedSpinLattice(SpinLattice):

    """
    Object representing a spin lattice system simulated by a trapped ion system
    
    Attributes:
        ti (TrappedIons): A trapped ion system
        mu (float or 1-D array of float): Raman beatnote detunings
        Omega (1-D or 2-D array of float): Rabi frequencies
    """

    def __init__(self, ti, mu, Omega, **kwargs):
        """
        Initialization function for a SimulatedSpinLattice object
        
        Args:
            ti (TrappedIons): A trapped ion system
            mu (float or 1-D array of float): Raman beatnote detunings
            Omega (1-D or 2-D array of float): Rabi frequencies
        Kwargs:
        """
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
        """SUMMARY
        
        Returns:
            TYPE: DESCRIPTION
        """
        eta = np.einsum(
            "in,n->in", self.b, 2 * self.k * np.sqrt(cst.hbar / (2 * self.m * self.w))
        )
        zeta = np.einsum("im,in->imn", self.Omega, eta)
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
        """SUMMARY
        
        Args:
            **kwargs: DESCRIPTION
        """
        pass

    def plot_rabi_frequencies(self, **kwargs):
        """SUMMARY
        
        Args:
            **kwargs: DESCRIPTION
        """
        pass

    pass
