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

########################################################################################

import pytest
from jax import numpy as jnp

from oqd_trical.mechanical.potential import (
    CoulombPotential,
    HarmonicPotential,
)

########################################################################################


def isequal(z, y):
    if not jnp.isclose(z, y).all():
        raise ValueError(f"{z.tolist()} is not equal target {y.tolist()}")


########################################################################################


class Test1DHarmonicPotential:
    @pytest.fixture
    def pot(self):
        return HarmonicPotential(
            [
                1,
            ]
        )

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (jnp.zeros((1, 1)), 0),
            (jnp.ones((1, 1)), 1),
            (jnp.ones((1, 1)) * 10, 100),
            (jnp.linspace(-1, 1, 2).reshape(-1, 1), 2),
            (jnp.linspace(-1, 1, 3).reshape(-1, 1), 2),
            (jnp.linspace(-5, 5, 11).reshape(-1, 1), 110),
        ],
    )
    def test_call(self, pot, x, y):
        isequal(pot(x), y)

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (jnp.zeros((1, 1)), 0),
            (jnp.ones((1, 1)), 2),
            (jnp.ones((1, 1)) * 10, 20),
            (jnp.linspace(-1, 1, 2).reshape(-1, 1), jnp.array([-2, 2]).reshape(-1, 1)),
            (
                jnp.linspace(-1, 1, 3).reshape(-1, 1),
                jnp.array([-2, 0, 2]).reshape(-1, 1),
            ),
            (
                jnp.linspace(-5, 5, 11).reshape(-1, 1),
                jnp.linspace(-10, 10, 11).reshape(-1, 1),
            ),
        ],
    )
    def test_grad(self, pot, x, y):
        isequal(pot.grad(x), y)

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (jnp.ones((1, 1)), 2),
            (jnp.ones((1, 1)), 2),
            (jnp.ones((1, 1)) * 10, 2),
            (
                jnp.linspace(-1, 1, 2).reshape(-1, 1),
                jnp.diag(2 * jnp.ones(2)).reshape(2, 1, 2, 1),
            ),
            # (jnp.linspace(-1, 1, 3).reshape(-1, 1), jnp.diag(2 * jnp.ones(3))),
            # (jnp.linspace(-5, 5, 11).reshape(-1, 1), jnp.diag(2 * jnp.ones(11))),
        ],
    )
    def test_hessian(self, pot, x, y):
        isequal(pot.hessian(x), y)


class Test2DHarmonicPotential:
    @pytest.fixture
    def pot(self):
        return HarmonicPotential([1, 5])

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (jnp.zeros((1, 2)), 0),
            (jnp.ones((1, 2)), 6),
            (jnp.array([0, 1]).reshape(-1, 2), 5),
            (jnp.array([0, 2]).reshape(-1, 2), 20),
            (
                jnp.concatenate(
                    [jnp.zeros((2, 1)), jnp.linspace(-1, 1, 2).reshape(-1, 1)], -1
                ),
                10,
            ),
            (
                jnp.concatenate(
                    [jnp.zeros((3, 1)), jnp.linspace(-1, 1, 3).reshape(-1, 1)], -1
                ),
                10,
            ),
            (
                jnp.concatenate(
                    [jnp.zeros((11, 1)), jnp.linspace(-5, 5, 11).reshape(-1, 1)], -1
                ),
                550,
            ),
        ],
    )
    def test_call(self, pot, x, y):
        isequal(pot(x), y)


class Test3DHarmonicPotential:
    @pytest.fixture
    def pot(self):
        return HarmonicPotential([1, 1, 5])

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (jnp.zeros((1, 3)), 0),
            (jnp.ones((1, 3)), 7),
            (jnp.array([0, 0, 1]).reshape(-1, 3), 5),
            (jnp.array([0, 0, 2]).reshape(-1, 3), 20),
            (
                jnp.concatenate(
                    [jnp.zeros((2, 2)), jnp.linspace(-1, 1, 2).reshape(-1, 1)], -1
                ),
                10,
            ),
            (
                jnp.concatenate(
                    [jnp.zeros((3, 2)), jnp.linspace(-1, 1, 3).reshape(-1, 1)], -1
                ),
                10,
            ),
            (
                jnp.concatenate(
                    [jnp.zeros((11, 2)), jnp.linspace(-5, 5, 11).reshape(-1, 1)], -1
                ),
                550,
            ),
        ],
    )
    def test_call(self, pot, x, y):
        isequal(pot(x), y)


########################################################################################


class TestCoulombPotential:
    @pytest.fixture
    def pot(self):
        return CoulombPotential()

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (jnp.zeros((1, 1)), 0),
            (jnp.ones((1, 1)), 0),
            (jnp.ones((1, 1)) * 10, 0),
            (jnp.linspace(-1, 1, 2).reshape(-1, 1), 0.5),
            (jnp.linspace(-1, 1, 3).reshape(-1, 1), 2.5),
            (
                jnp.linspace(-5, 5, 11).reshape(-1, 1),
                (jnp.arange(1, 11) / jnp.arange(10, 0, -1)).sum(),
            ),
            (jnp.zeros((1, 2)), 0),
            (jnp.ones((1, 2)), 0),
            (jnp.array([0, 1]).reshape(-1, 2), 0),
            (jnp.array([0, 2]).reshape(-1, 2), 0),
            (
                jnp.concatenate(
                    [jnp.zeros((2, 1)), jnp.linspace(-1, 1, 2).reshape(-1, 1)], -1
                ),
                0.5,
            ),
            (
                jnp.concatenate(
                    [jnp.zeros((3, 1)), jnp.linspace(-1, 1, 3).reshape(-1, 1)], -1
                ),
                2.5,
            ),
            (
                jnp.concatenate(
                    [jnp.zeros((11, 1)), jnp.linspace(-5, 5, 11).reshape(-1, 1)], -1
                ),
                (jnp.arange(1, 11) / jnp.arange(10, 0, -1)).sum(),
            ),
            (jnp.zeros((1, 3)), 0),
            (jnp.ones((1, 3)), 0),
            (jnp.array([0, 0, 1]).reshape(-1, 3), 0),
            (jnp.array([0, 0, 2]).reshape(-1, 3), 0),
            (
                jnp.concatenate(
                    [jnp.zeros((2, 2)), jnp.linspace(-1, 1, 2).reshape(-1, 1)], -1
                ),
                0.5,
            ),
            (
                jnp.concatenate(
                    [jnp.zeros((3, 2)), jnp.linspace(-1, 1, 3).reshape(-1, 1)], -1
                ),
                2.5,
            ),
            (
                jnp.concatenate(
                    [jnp.zeros((11, 2)), jnp.linspace(-5, 5, 11).reshape(-1, 1)], -1
                ),
                (jnp.arange(1, 11) / jnp.arange(10, 0, -1)).sum(),
            ),
        ],
    )
    def test_call(self, pot, x, y):
        isequal(pot(x), y)

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (jnp.zeros((1, 1)), 0),
            (jnp.ones((1, 1)), 0),
            (jnp.ones((1, 1)) * 10, 0),
            (
                jnp.linspace(-1, 1, 2).reshape(-1, 1),
                jnp.array([0.25, -0.25]).reshape(-1, 1),
            ),
            (
                jnp.linspace(-1, 1, 3).reshape(-1, 1),
                jnp.array([1.25, 0, -1.25]).reshape(-1, 1),
            ),
            (
                jnp.linspace(-5, 5, 11).reshape(-1, 1),
                jnp.array(
                    [
                        1.5497677326202393,
                        0.5397677421569824,
                        0.27742207050323486,
                        0.1506861448287964,
                        0.06777727603912354,
                        0.0,
                        -0.06777715682983398,
                        -0.1506861448287964,
                        -0.27742207050323486,
                        -0.5397677421569824,
                        -1.5497677326202393,
                    ]
                ).reshape(-1, 1),
            ),
        ],
    )
    def test_grad(self, pot, x, y):
        isequal(pot.grad(x), y)

    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (jnp.ones((1, 1)), 0),
            (jnp.ones((1, 1)), 0),
            (jnp.ones((1, 1)) * 10, 0),
            (
                jnp.linspace(-1, 1, 2).reshape(-1, 1),
                jnp.array([[0.25, -0.25], [-0.25, 0.25]]).reshape(2, 1, 2, 1),
            ),
            (
                jnp.linspace(-1, 1, 3).reshape(-1, 1),
                jnp.array(
                    [[2.25, -2.0, -0.25], [-2.0, 4.0, -2.0], [-0.25, -2.0, 2.25]]
                ).reshape(3, 1, 3, 1),
            ),
        ],
    )
    def test_hessian(self, pot, x, y):
        isequal(pot.hessian(x), y)
