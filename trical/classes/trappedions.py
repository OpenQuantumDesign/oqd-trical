from . import constants as cst
from .potential import CoulombPotential
from matplotlib import pyplot as plt
import numpy as np


class TrappedIons(object):
    def __init__(self, N, *ps, **kwargs):
        super(IonChain, self).__init__()

        params = {"l": 1e-6, "m": cst.m_a["Yb171"], "q": cst.e}
        params.update(kwargs)
        self.__dict__.update(params)

        self.N = N
        self.ps = np.array(ps)

        self.cp = CoulombPotential(N, q=self.q)
        self.fp = self.cp + self.ps.sum()
        pass

    def equilibrium_position(self, opt):
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.apply_along_axis(lambda p: p.nondimensionalize(), self.ps)
        ndfp = self.ndcp + self.ndps.sum()

        self.x_ep = opt(ndfp) * self.l
        return self.x_ep

    def normal_modes(self):
        pass

    pass
