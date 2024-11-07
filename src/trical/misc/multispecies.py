"""
Module containing useful functions relavent for multi-species systems.
"""

########################################################################################

import numpy as np

########################################################################################

"""
Module containing useful functions and classes relevant for multi-species systems.
"""

import numpy as np
from .polynomial import PolynomialPotential

########################################################################################

def dc_trap_geometry(omega):
    """
    Calculates the trap geometry of a trapped ion system.

    Args:
        omega (np.ndarray[float]): Trap strengths of primary species

    Returns:
        (np.ndarray[float]): Trap geometry factors of the system
    """
    gamma_diff = (omega[1] ** 2 - omega[0] ** 2) / omega[2] ** 2
    gamma_x = (gamma_diff + 1) / 2
    gamma_y = 1 - gamma_x
    gamma = np.array([gamma_x, gamma_y, 1])
    return gamma


def ms_trap_strength(m, m0, omega):
    """
    Calculates the transverse trap frequencies of non-primary species.

    Args:
        m (np.ndarray[float]): Mass of ions
        m0 (float): Mass of primary species
        omega (np.ndarray[float]): Trap strengths of primary species

    Returns:
        (np.ndarray[float]): Trap strengths of the ions
    """
    omega_dc = omega[2]

    gamma = dc_trap_geometry(omega)[:2]
    omega_rf = np.sqrt(omega[0] ** 2 + omega_dc**2 * gamma[0])

    omega_axial = np.sqrt(m0 / m) * omega[2]
    omega_trans = np.sqrt(
        (m0 / m) ** 2 * omega_rf**2 - np.outer(gamma, (m0 / m)) * omega_dc**2
    )

    omegas = np.vstack((omega_trans, omega_axial))
    return omegas


class TrappedIons:
    """
    Class to represent a system of trapped ions, including multi-species systems.

    Attributes:
        N (int): Number of ions in the trap.
        potential (PolynomialPotential): The polynomial potential for the ions.
        mass (float): Mass of the ions.
    """

    def __init__(self, N: int, potential: PolynomialPotential, m: float):
        self.N = N
        self.potential = potential
        self.mass = m
        self.w_pa = None  # eigenfreqs
        self.b_pa = None  # eigenvecs

    def principle_axis(self):
        """
        Calculate the principal axes and eigenfrequencies of the trapped ions.
        This is a placeholder function and should be replaced with the actual calculation.
        """
        # calculation for eigenfrequencies and eigenvectors computation here
        self.w_pa = np.random.rand(self.N)  # eigenfreqs
        self.b_pa = np.random.rand(self.N, self.N)  # eigenvecs

    def equilibrium_position(self):
        """
        Calculate equilibrium positions of the ions.

        Returns:
            np.ndarray: Equilibrium positions of the ions.
        """
        # replace with the actual equilibrium calculation
        return np.linspace(-1, 1, self.N)


def dc_trap_geometry(omega):
    """
    Calculates the trap geometry of a trapped ion system.

    Args:
        omega (np.ndarray[float]): Trap strengths of primary species

    Returns:
        (np.ndarray[float]): Trap geometry factors of the system
    """
    gamma_diff = (omega[1] ** 2 - omega[0] ** 2) / omega[2] ** 2
    gamma_x = (gamma_diff + 1) / 2
    gamma_y = 1 - gamma_x
    gamma = np.array([gamma_x, gamma_y, 1])
    return gamma


def ms_trap_strength(m, m0, omega):
    """
    Calculates the transverse trap frequencies of non-primary species.

    Args:
        m (np.ndarray[float]): Mass of ions
        m0 (float): Mass of primary species
        omega (np.ndarray[float]): Trap strengths of primary species

    Returns:
        (np.ndarray[float]): Trap strengths of the ions
    """
    omega_dc = omega[2]

    gamma = dc_trap_geometry(omega)[:2]
    omega_rf = np.sqrt(omega[0] ** 2 + omega_dc**2 * gamma[0])

    omega_axial = np.sqrt(m0 / m) * omega[2]
    omega_trans = np.sqrt(
        (m0 / m) ** 2 * omega_rf**2 - np.outer(gamma, (m0 / m)) * omega_dc**2
    )

    omegas = np.vstack((omega_trans, omega_axial))
    return omegas
