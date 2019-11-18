from .. import constants as cst
from ..misc.optimize import dflt_opt
from .potential import CoulombPotential
from matplotlib import pyplot as plt
import numpy as np


class TrappedIons(object):
    def __init__(self, N, *ps, **kwargs):
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
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        _ndfp = lambda x: ndfp(x.reshape(self.dim, self.N).transpose())

        self.x_ep = (
            opt(self.N, self.dim)(_ndfp).reshape(self.dim, self.N).transpose() * self.l
        )
        return self.x_ep

    def normal_modes(self):
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        hess_phi = ndfp.hessian()

        self.equilibrium_position()

        hess_phi_x_ep = hess_phi(self.x_ep / self.l)

        self.w, self.b = np.linalg.eig(hess_phi_x_ep)
        return self.w, self.b

    pass
