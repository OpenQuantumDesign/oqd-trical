from typing import Callable

import torch
from torch.autograd import grad
from torch.autograd.functional import hessian

from functools import cached_property

from abc import ABC, abstractmethod

########################################################################################


class PotentialBase(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def phi(self):
        pass

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


class Potential(PotentialBase):
    def __init__(self, phi: Callable):
        self._phi = phi
        pass

    @property
    def phi(self):
        return self._phi


class CoulombPotential(PotentialBase):
    def __init__(self):
        super().__init__()

    @cached_property
    def phi(self):
        def _phi(x):
            N = x.shape[-1]

            idcs = torch.triu_indices(N, N, 1)
            x1 = x[:, idcs[0]]
            x2 = x[:, idcs[1]]
            y = (2 / torch.norm(x1 - x2, dim=0)).sum(0)

            return y

        return _phi


class HarmonicPotential(PotentialBase):
    def __init__(self):
        super().__init__()

    @cached_property
    def phi(self):
        def _phi(x):
            return (
                0.5 * (2 * torch.pi * torch.tensor([1, 1, 10])[:, None] * x) ** 2
            ).sum()

        return _phi


class PolynomialPotential(PotentialBase):
    def __init__(self):
        raise NotImplementedError
        super().__init__()
        pass


class OpticalPotential(PotentialBase):
    def __init__(self):
        raise NotImplementedError
        super().__init__()
        pass
