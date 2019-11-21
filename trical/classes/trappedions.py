"""
Defines the TrappedIons class representing a trapped ions system
"""
from .. import constants as cst
from ..misc.linalg import norm, orthonormal_subset
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

        idcs = np.argsort(w)

        self.w, self.b = w[idcs], b[:, idcs]
        return self.w, self.b

    def principle_axis(self, tol=1e-3):
        if "w" not in self.__dict__.keys() or "b" not in self.__dict__.keys():
            self.normal_modes()

        x_pa = orthonormal_subset(self.b.reshape(self.dim, -1).transpose(), tol=tol)

        assert len(x_pa) == self.dim

        b_pa = np.einsum("ij,jn->in", x_pa, self.b.reshape(self.dim, -1)).reshape(
            self.dim * self.N, -1
        )
        w_pa = self.w

        n = np.array(
            np.round(
                [
                    norm(b_pa[i * self.N : (i + 1) * self.N].transpose())
                    for i in range(self.dim)
                ]
            )
        )
        idcs = np.lexsort(np.concatenate((w_pa.reshape(1, -1), n)))

        self.x_pa, self.w_pa, self.b_pa = x_pa, w_pa[idcs], b_pa[:, idcs]
        return self.x_pa, self.w_pa, self.b_pa

    pass
