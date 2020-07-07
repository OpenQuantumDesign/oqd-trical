import numpy as np

"""
Module containing useful functions relavent for multi-species systems.
"""

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
