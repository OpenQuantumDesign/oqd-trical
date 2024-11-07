"""
Module containing relevant constants, in SI units, for TrICal.

Attributes:
    e (float): Elementary charge.
    hbar (float): Reduced Planck constant.
    k (float): Coulomb constant.
    c (float): Speed of light.
"""

########################################################################################

import numpy as np

########################################################################################

c = 2.99792458e8
e = 1.602176634e-19
hbar = 1.054571817e-34
k = 8.9875517923e9
eps_0 = 8.854187817e-12 

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
    return (2 * k * q**2 / (m * omega**2)) ** (1 / 3)


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
    return k * q**2 / natural_l(m, q, omega)
