from typing import Callable

import torch
from torch.autograd import grad
from torch.autograd.functional import hessian

from functools import cached_property

########################################################################################


class Potential:
    def __init__(self, phi: Callable):
        self._phi = phi
        pass

    @property
    def phi(self):
        return self._phi

    def __call__(self, x: torch.Tensor):
        return self.phi(x)

    def gradient(self, x: torch.Tensor):
        x.requires_grad_(True)
        return grad(self(x), x, create_graph=True)

    def hessian(self, x: torch.Tensor):
        x.requires_grad_(True)
        return hessian(self, x, create_graph=True)

    def __add__(self, other):
        return Potential(lambda x: self(x) + other(x))


class CoulombPotential(Potential):
    def __init__(self):
        super().__init__(self.phi_factory())

    def phi_factory(self):
        def _phi(x):
            N = x.shape[-1]

            idcs = torch.triu_indices(N, N, 1)
            x1 = x[:, idcs[0]]
            x2 = x[:, idcs[1]]
            y = (2 / torch.norm(x1 - x2, dim=0)).sum(0)

            return y

        return _phi


class HarmonicPotential(Potential):
    def __init__(self):
        raise NotImplementedError
        super().__init__()
        pass


class PolynomialPotential(Potential):
    def __init__(self):
        raise NotImplementedError
        super().__init__()
        pass


class OpticalPotential(Potential):
    def __init__(self):
        raise NotImplementedError
        super().__init__()
        pass
