from .potential import CoulombPotential
import numpy as np


class TrappedIons(object):
    def __init__(self, N, *ps, **kwargs):
        super(IonChain, self).__init__()

        self.N = N
        self.cp = CoulombPotential(N)
        self.ps = ps

        params = {}
        params.update(kwargs)
        self.__dict__.update(params)
        pass

    def equilibrium_position(self, opt):
        pass

    def normal_modes(self):
        pass

    pass
