from . import constants as cst
from .potential import CoulombPotential
from matplotlib import pyplot as plt
import numpy as np


class TrappedIons(object):
    def __init__(self, N, *ps, **kwargs):
        super(IonChain, self).__init__()

        params = {"m": cst.m_a["Yb171"], "q": cst.e}
        params.update(kwargs)
        self.__dict__.update(params)

        self.N = N
        self.ps = np.array(ps)

        self.cp = CoulombPotential(N, q=self.q)
        self.fp = self.cp + self.ps.sum()
        pass

    def equilibrium_position(self, opt):
        return opt(self.fp)

    def normal_modes(self):
        pass

    pass
