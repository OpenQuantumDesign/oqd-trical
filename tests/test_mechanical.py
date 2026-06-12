# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import product

import numpy as np
import pytest
import sympy as sym
from autograd import numpy as agnp

from oqd_trical.mechanical import (
    AutoDiffPotential,
    CoulombPotential,
    GaussianOpticalPotential,
    PolynomialPotential,
    SymbolicPotential,
)
from oqd_trical.misc import constants as cst

########################################################################################


class TestPotential:
    @pytest.fixture
    def configuration(self, request):
        dim, N = request.param

        x = np.zeros((N, dim))
        x[:, -1] = np.arange(N) - (N - 1) / 2

        return x

    @pytest.mark.parametrize(
        "configuration",
        product(np.arange(1, 4, dtype=int), np.logspace(0, 5, 6, base=2, dtype=int)),
        indirect=True,
    )
    def test_polynomial_potential(self, configuration):
        N, dim = configuration.shape

        alpha = np.zeros([3] * dim)
        alpha[tuple(np.eye(dim, dtype=int) * 2)] = np.array([*([1] * (dim - 1)), 5])

        potential = PolynomialPotential(alpha, N=N)
        potential(configuration)
        potential.gradient()(configuration)
        potential.hessian()(configuration)

    @pytest.mark.parametrize(
        "configuration",
        product(np.arange(1, 4, dtype=int), np.logspace(0, 5, 6, base=2, dtype=int)),
        indirect=True,
    )
    def test_coulomb_potential(self, configuration):
        N, dim = configuration.shape

        potential = CoulombPotential(N=N, dim=dim)
        potential(configuration)
        potential.gradient()(configuration)
        potential.hessian()(configuration)

    @pytest.mark.parametrize(
        "configuration",
        product(np.arange(1, 4, dtype=int), np.logspace(0, 5, 6, base=2, dtype=int)),
        indirect=True,
    )
    def test_autodiff_potential(self, configuration):
        N, dim = configuration.shape

        def expr(x):
            return agnp.sum(agnp.array([*([1] * (dim - 1)), 5])[None, :] * x**2)

        potential = AutoDiffPotential(expr=expr, N=N, dim=dim)
        potential(configuration)
        potential.gradient()(configuration)
        potential.hessian()(configuration)

    @pytest.mark.parametrize(
        "configuration",
        product(np.arange(1, 4, dtype=int), np.logspace(0, 5, 6, base=2, dtype=int)),
        indirect=True,
    )
    def test_symbolic_potential(self, configuration):
        N, dim = configuration.shape

        symbols = sym.symbols("x,y,z")
        x = symbols[:dim]

        expr = np.sum(np.array([*([1] * (dim - 1)), 5]) * np.array(x) ** 2)

        potential = SymbolicPotential(expr=expr, N=N, dim=dim)
        potential(configuration)
        potential.gradient()(configuration)
        potential.hessian()(configuration)

    @pytest.mark.parametrize(
        "configuration",
        product(np.arange(3, 4, dtype=int), np.logspace(0, 5, 6, base=2, dtype=int)),
        indirect=True,
    )
    def test_gaussian_optical_potential(self, configuration):
        N, dim = configuration.shape

        mass = 171 * cst.m_u
        focal_point = np.zeros(3)
        beam_waist = 1e-6
        refractive_index = 1.0
        wavelength = 375e-9
        power = 1e0
        Omega_bar = 2.23e6
        transition_wavelength = 369.52e-9

        potential = GaussianOpticalPotential(
            focal_point,
            power,
            wavelength,
            beam_waist,
            mass=mass,
            Omega_bar=Omega_bar,
            transition_wavelength=transition_wavelength,
            refractive_index=refractive_index,
            N=N,
            dim=dim,
        )
        potential(configuration)
        potential.gradient()(configuration)
        potential.hessian()(configuration)


########################################################################################


class TestTrappedIons:
    pass


class TestSpinLattice:
    pass
