from .. import constants as cst
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

        self.cp = CoulombPotential(N, q=self.q)
        self.fp = self.cp + self.ps.sum()
        pass

    def equilibrium_position(self, opt):
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()
        pass

    def normal_modes(self):
        pass

    pass
