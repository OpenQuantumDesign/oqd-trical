import numpy as np

"""
Module containing useful functions relavent for multi-species systems.
"""


def dc_trap_geometry(omega):
    """
    Calculates the trap geometry of a trapped ion system.

    :param omega: Trap strengths of primary species
    :type omega: :obj:`numpy.ndarray`
    :returns: Trap geometry factors of the system
    :rtype: :obj:`numpy.ndarray`
    """
    gamma_diff = (omega[1] ** 2 - omega[0] ** 2) / omega[2] ** 2
    gamma_x = (gamma_diff + 1) / 2
    gamma_y = 1 - gamma_x
    gamma = np.array([gamma_x, gamma_y, 1])
    return gamma


def ms_trap_strength(m, m0, omega):
    """
    Calculates the transverse trap frequencies of non-primary species.

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

    gamma = dc_trap_geometry(omega)[:2]
    omega_rf = np.sqrt(omega[0] ** 2 + omega_dc ** 2 * gamma[0])

    omega_axial = np.sqrt(m0 / m) * omega[2]
    omega_trans = np.sqrt(
        (m0 / m) ** 2 * omega_rf ** 2 - np.outer(gamma, (m0 / m)) * omega_dc ** 2
    )

    omegas = np.vstack((omega_trans, omega_axial))
    return omegas
