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

########################################################################################


def convert_m_a(A):
    """
    Converts atomic mass from atomic mass units to kilograms

    Args:
        A (float): Atomic mass in atomic mass units

    Returns:
        (float): Atomic mass in kilograms
    """
    return A * 1.66053906660e-27


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
