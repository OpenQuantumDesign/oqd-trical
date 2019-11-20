"""
Defines the TrappedIons class representing a trapped ions system
"""
from .. import constants as cst
from ..misc.linalg import norm
from ..misc.optimize import dflt_opt
from .potential import CoulombPotential
from matplotlib import pyplot as plt
import numpy as np


class TrappedIons(object):

    """
    Object representing a system of trapped ions
    
    Attributes:
        b (2-D array of float): Normal mode eigenvectors of the system
        cp (CoulombPotential): Coulomb potential associated with the system
        dim (int): Dimension of the system
        fp (Potential): Total potential of the system
        l (float): Length scale of the system
        m (float): Mass of an ion
        N (int): Number of ions
        ps (array of Potential): Non-Coulomb potentials associated with the system
        q (float): Charge of an ion
        x_ep (1D or 2-D array of float): Equilibrium position of the ions
        w (1-D array of float): Normal mode frequencies of the system
    """

    def __init__(self, N, *ps, **kwargs):
        """
        Initialization function for a TrappedIons object
        
        Args:
            N (int): Number of Ions
            ps (array of Potential): Non-Coulomb potentials associated with the system
        
        Kwargs:
            dim (int, optional): Dimension of the system
            l (float, optional): Length scale of the system
            m (float, optional): Mass of an ion
            q (float, optional): Charge of an ion
        """
        super(TrappedIons, self).__init__()

        params = {
            "dim": ps[0].dim if "dim" in ps[0].__dict__.keys() else 3,
            "l": 1e-6,
            "m": cst.m_a["Yb171"],
            "q": cst.e,
        }
        params.update(kwargs)
        self.__dict__.update(params)

        self.N = N
        self.ps = np.array(ps)

        self.cp = CoulombPotential(N, dim=self.dim, q=self.q)
        self.fp = self.cp + self.ps.sum()
        pass

    def equilibrium_position(self, opt=dflt_opt):
        """
        Function that calculates the equilibrium position of the ions
        
        Args:
            opt (TYPE, optional): DESCRIPTION
            opt (func(TrappedIons) -> func(func(1-D array of float)-> 1-D array of float)
            , optional): Generator of an optimization function that minimizes the
            potential of the system with respect to the position of the ions
        
        Returns:
            1D or 2-D array of float: Equilibrium position of the ions
        """
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        _ndfp = lambda x: ndfp(x.reshape(self.dim, self.N).transpose())

        self.x_ep = opt(self)(_ndfp).reshape(self.dim, self.N).transpose() * self.l
        return self.x_ep

    def normal_modes(self):
        """
        Function that calculates the normal modes of the system
        
        Returns:
            1-D array of float: Normal mode frequencies of the system
            2-D array of float: Normal mode eigenvectors of the system

        """
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        hess_phi = ndfp.hessian()

        if "x_ep" not in self.__dict__.keys():
            self.equilibrium_position()

        hess_phi_x_ep = hess_phi(self.x_ep / self.l)

        w, b = np.linalg.eigh(hess_phi_x_ep)
        w = np.sqrt(w * cst.k * cst.e ** 2 / (self.m * self.l ** 3))

        idcs = np.lexsort(
            np.concatenate(
                (
                    w.reshape(1, -1),
                    np.array(
                        [
                            np.round(norm(b[i * self.N : (i + 1) * self.N].transpose()))
                            for i in range(self.dim)
                        ]
                    ),
                )
            )
        )

        self.w, self.b = w[idcs], b[:, idcs]
        return self.w, self.b

    pass
