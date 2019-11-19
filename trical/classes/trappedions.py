"""SUMMARY
"""
from .. import constants as cst
from ..misc.linalg import norm
from ..misc.optimize import dflt_opt
from .potential import CoulombPotential
from matplotlib import pyplot as plt
import numpy as np


class TrappedIons(object):

    """SUMMARY
    
    Attributes:
        cp (TYPE): DESCRIPTION
        fp (TYPE): DESCRIPTION
        N (TYPE): DESCRIPTION
        ps (TYPE): DESCRIPTION
        x_ep (TYPE): DESCRIPTION
    """
    
    def __init__(self, N, *ps, **kwargs):
        """SUMMARY
        
        Args:
            N (TYPE): DESCRIPTION
            *ps: DESCRIPTION
            **kwargs: DESCRIPTION
        """
        super(TrappedIons, self).__init__()

        params = {"dim": 3, "l": 1e-6, "m": cst.m_a["Yb171"], "q": cst.e}
        params.update(kwargs)
        self.__dict__.update(params)

        self.N = N
        self.ps = np.array(ps)

        self.cp = CoulombPotential(N, dim=self.dim, q=self.q)
        self.fp = self.cp + self.ps.sum()
        pass

    def equilibrium_position(self, opt=dflt_opt):
        """SUMMARY
        
        Args:
            opt (TYPE, optional): DESCRIPTION
        
        Returns:
            TYPE: DESCRIPTION
        """
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        _ndfp = lambda x: ndfp(x.reshape(self.dim, self.N).transpose())

        self.x_ep = (
            opt(self.N, self.dim)(_ndfp).reshape(self.dim, self.N).transpose() * self.l
        )
        return self.x_ep

    def normal_modes(self):
        """SUMMARY
        
        Returns:
            TYPE: DESCRIPTION
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
