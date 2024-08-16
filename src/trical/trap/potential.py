from typing import Callable

import torch
from torch.autograd import grad
from torch.autograd.functional import hessian

########################################################################################


class Potential:
    def __init__(self, phi: Callable):
        self.phi = phi
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


class CoulombPotential(Potential):
    def __init__(self):
        raise NotImplementedError
        super().__init__()
        pass


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
