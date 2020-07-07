import numpy as np

"""
Module containing relevant constants, in SI units, for TrICal.

:Variables:
    * **e** (:obj:`float`): Elementary charge.
    * **hbar** (:obj:`float`): Reduced Planck constant.
    * **k** (:obj:`float`): Coulomb constant.
    * **m_a** (:obj:`dict`): Dictionary of atomic masses.
"""

c = 2.99792458e8
e = 1.602176634e-19
hbar = 1.054571817e-34
k = 8.9875517923e9


def convert_m_a(A):
    """
    Converts atomic mass from atomic mass units to kilograms

    :param A: Atomic mass in atomic mass units
    :type A: :obj:`float`
    :returns: Atomic mass in kilograms
    :rtype: :obj:`float`
    """
    return A * 1.66053906660e-27


def convert_lamb_to_omega(lamb):
    return 2 * np.pi * c / lamb


def natural_l(m, q, omega):
    """
    Calculates a natural length scale for a trapped ion system

    :param m: Mass of ion
    :type m: :obj:`float`
    :param q: Charge of ion
    :type q: :obj:`float`
    :param omega: Trapping strength
    :type omega: :obj:`float`
    :returns: Natural length scale
    :rtype: :obj:`float`
    """
    return (2 * k * q ** 2 / (m * omega ** 2)) ** (1 / 3)


def natural_V(m, q, omega):
    """
    Calculates a natural energy scale for a trapped ion system

    :param m: Mass of ion
    :type m: :obj:`float`
    :param q: Charge of ion
    :type q: :obj:`float`
    :param omega: Trapping strength
    :type omega: :obj:`float`
    :returns: Natural energy scale
    :rtype: :obj:`float`
    """
    return k * q ** 2 / natural_l(m, q, omega)


def ms_trap_strength(m, m0, omega):
    """
    Calculates the transverse trap frequencies of non-primary species

    :param m: Mass of ions
    :type m: :obj:`numpy.ndarray`
    :param m0: Mass of primary species
    :type m0: :obj:`float`
    :param omega: Trap strengths of primary species
    :type omega: :obj:`numpy.ndarray`
    :returns: Trap strengths of the ions
    :rtype: :obj:`numpy.ndarray`
    """
    omega_dc = omega[2]

    omega_axial = np.sqrt(m / m0) * omega[2]

    gamma_diff = (omega[1] ** 2 - omega[0] ** 2) / omega_dc ** 2
    gamma_x = (gamma_diff + 1) / 2
    gamma_y = 1 - gamma_x
    gamma = np.array([gamma_x, gamma_y])

    omega_rf = np.sqrt(omega[:2] ** 2 + omega_dc ** 2 * gamma)

    omega_trans = np.sqrt(
        (m0 / m) ** 2 * omega_rf ** 2 - np.outer(gamma, (m0 / m)) * omega_dc ** 2
    )

    omegas = np.hstack((omega_trans), omega_axial)
    return omegas
