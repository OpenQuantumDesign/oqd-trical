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

from oqd_compiler_infrastructure import Chain

from oqd_trical.light_matter.compiler.canonicalize import (
    canonicalize_coefficient_factory,
    canonicalize_math_factory,
)
from oqd_trical.light_matter.interface import ConstantCoefficient, WaveCoefficient

########################################################################################


def test_simplify_sum_of_constants():
    a = ConstantCoefficient(value=1)
    b = ConstantCoefficient(value=2)

    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())
    c = simplify(a + b)
    assert c == ConstantCoefficient(value=3)


def test_simplify_product_of_constants():
    a = ConstantCoefficient(value=1)
    b = ConstantCoefficient(value=2)

    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())
    c = simplify(a * b)
    assert c == ConstantCoefficient(value=2)


def test_simplify_sum_of_waves():
    a = WaveCoefficient(amplitude=1, frequency=10, phase=100)
    b = WaveCoefficient(amplitude=2, frequency=20, phase=200)

    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())
    c = simplify(a + b)
    assert c == a + b


def test_simplify_product_of_waves():
    a = WaveCoefficient(amplitude=1, frequency=10, phase=100)
    b = WaveCoefficient(amplitude=2, frequency=20, phase=200)

    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())
    c = simplify(a * b)
    assert c == WaveCoefficient(amplitude=2, frequency=30, phase=300)


########################################################################################


def test_division_constants():
    a = ConstantCoefficient(value=1)
    b = ConstantCoefficient(value=2)

    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())
    c = simplify(a / b)
    assert c == ConstantCoefficient(value=1 / 2)


def test_division_waves():
    a = WaveCoefficient(amplitude=1, frequency=10, phase=100)
    b = WaveCoefficient(amplitude=2, frequency=20, phase=200)

    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())
    c = simplify(a / b)
    assert c == WaveCoefficient(amplitude=1 / 2, frequency=-10, phase=-100)


def test_division_of_sum_of_waves():
    a = WaveCoefficient(amplitude=1, frequency=10, phase=100) + WaveCoefficient(
        amplitude=2, frequency=20, phase=200
    )
    b = WaveCoefficient(amplitude=3, frequency=30, phase=300)

    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())
    c = simplify(a / b)
    assert c == (
        WaveCoefficient(amplitude=1 / 3, frequency=-20, phase=-200)
        + WaveCoefficient(amplitude=2 / 3, frequency=-10, phase=-100)
    )


def test_division_of_product_of_waves():
    a = WaveCoefficient(amplitude=1, frequency=10, phase=100) * WaveCoefficient(
        amplitude=2, frequency=20, phase=200
    )
    b = WaveCoefficient(amplitude=3, frequency=30, phase=300)

    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())
    c = simplify(a / b)
    assert c == ConstantCoefficient(value=2 / 3)


########################################################################################


def test_conjugate_real():
    a = ConstantCoefficient(value=1)
    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())

    b = simplify(a.conj())
    assert b == a


def test_conjugate_wave():
    a = WaveCoefficient(amplitude=1, frequency=10, phase=100)
    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())

    b = simplify(a.conj())
    assert b == WaveCoefficient(amplitude=1, frequency=-10, phase=-100)


def test_conjugate_of_sum_of_waves():
    a = WaveCoefficient(amplitude=1, frequency=10, phase=100) + WaveCoefficient(
        amplitude=2, frequency=20, phase=200
    )
    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())

    b = simplify(a.conj())
    assert b == (
        WaveCoefficient(amplitude=1, frequency=-10, phase=-100)
        + WaveCoefficient(amplitude=2, frequency=-20, phase=-200)
    )


def test_conjugate_of_product_of_waves():
    a = WaveCoefficient(amplitude=1, frequency=10, phase=100) * WaveCoefficient(
        amplitude=2, frequency=20, phase=200
    )
    simplify = Chain(canonicalize_coefficient_factory(), canonicalize_math_factory())

    b = simplify(a.conj())
    assert b == (WaveCoefficient(amplitude=2, frequency=-30, phase=-300))
