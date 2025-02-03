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

"""
Module containing relevant constants, in SI units, for TrICal
"""

########################################################################################

import numpy as np

########################################################################################

c: float = 2.99792458e8
"""
Speed of light
"""
e: float = 1.602176634e-19
"""
Elementary charge
"""
hbar: float = 1.054571817e-34
"""
Reduced Planck constant
"""
k_e: float = 8.9875517923e9
"""
Coulomb constant
"""
epsilon_0: float = 8.8541878188e-12
"""
Permittivity of free space
"""
m_u: float = 1.66053906892e-27
"""
Atomic mass unit
"""

########################################################################################


def convert_m_a(A):
    """
    Converts atomic mass from atomic mass units to kilograms

    Args:
        A (float): Atomic mass in atomic mass units

    Returns:
        (float): Atomic mass in kilograms
    """
    return A * m_u


def convert_lamb_to_omega(lamb):
    return 2 * np.pi * c / lamb


def natural_l(m, q, omega):
    """
    Calculates a natural length scale for a trapped ion system

    Args:
        m (float): Mass of ion
        q (float): Charge of ion
        omega (float): Trapping strength

    Returns:
        (float): Natural length scale
    """
    return (2 * k_e * q**2 / (m * omega**2)) ** (1 / 3)


def natural_V(m, q, omega):
    """
    Calculates a natural energy scale for a trapped ion system

    Args:
        m (float): Mass of ion
        q (float): Charge of ion
        omega (float): Trapping strength

    Returns:
        (float): Natural energy scale
    """
    return k_e * q**2 / natural_l(m, q, omega)
