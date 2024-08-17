from functools import cached_property

import torch

########################################################################################

from .optimizer import TorchOptimizer

########################################################################################


class IonCrystal:
    def __init__(self, potential, *, N=10, dim=3, optimizer=TorchOptimizer()):
        self.potential = potential
        self.dim = dim
        self.N = N
        self.optimizer = optimizer
        pass

    @cached_property
    def equilibrium_position(self):
        parameters = torch.nn.Parameter(torch.empty(self.dim, self.N))

        return self.optimizer.optimize(self.potential, parameters)

    @cached_property
    def normal_modes(self):
        A = self.potential.hessian(self.equilibrium_position)
        A = A.reshape(self.dim * self.N, self.dim * self.N)
        E = torch.linalg.eigh(A)
        return E[0], E[1].reshape(self.dim, self.N, self.dim * self.N)
