import numpy as np

from sympy.physics.wigner import wigner_3j, wigner_6j

########################################################################################

from ..misc import constants as cst

########################################################################################


def compute_matrix_element(laser, transition):
    """Method that compute dipole and quadrupole matrix elements

    Args:
        laser (Laser): laser object for accessing polarization and wavevector information
        transition (Transition): transition object for accessing quantum number of levels

    Returns:
        (float): Multipole matrix elements for E1, E2 transitions
    """

    # If this happens there's probably an error with the ion species card
    if transition.level1.nuclear != transition.level2.nuclear:
        raise ValueError(
            "Different nuclear spins between two levels in transition:", transition
        )

    # Just by convention; these orderings are set upon instantiation of an Ion object
    if transition.level1.energy > transition.level2.energy:
        raise ValueError("Expected energy of level2 > energy of level1")

    polarization = np.array(laser.polarization).T  # make it a column vector
    wavevector = laser.wavevector

    J1, J2, F1, F2, M1, M2, E1, E2, I, A = (
        transition.level1.spin_orbital,
        transition.level2.spin_orbital,
        transition.level1.spin_orbital_nuclear,
        transition.level2.spin_orbital_nuclear,
        transition.level1.spin_orbital_nuclear_magnetization,
        transition.level2.spin_orbital_nuclear_magnetization,
        transition.level1.energy,
        transition.level2.energy,
        transition.level1.nuclear,
        transition.einsteinA,
    )

    q = M2 - M1

    omega_0 = E2 - E1

    # TODO M1 transition multipole

    if transition.multipole == "M1":
        pass

    if transition.multipole == "E1":

        units_term = np.sqrt(
            (2 * np.pi * cst.epsilon_0 * cst.hbar * cst.c**3) / (omega_0 * A)
        ) / (omega_0 * cst.e)
        hyperfine_term = np.sqrt((2 * F2 + 1) * (2 * F1 + 1)) * wigner_6j(
            J1, J2, 1, F2, F1, I
        )

        # q -> polarization
        polarization_map = {
            -1: 1 / np.sqrt(2) * np.array([1, 1j, 0]),
            0: np.array([0, 0, 1]),
            1: 1 / np.sqrt(2) * np.array([1, -1j, 0]),
        }

        geometry_term = (
            np.sqrt(2 * J2 + 1)
            * polarization_map[q].dot(polarization)
            * wigner_3j(F2, 1, F1, M2, -q, -M1)
        )

        return float(
            (abs(units_term) * abs(geometry_term) * abs(hyperfine_term)).evalf()
        )

    elif transition.multipole == "E2":

        units_term = np.sqrt(
            (15 * np.pi * cst.epsilon_0 * cst.hbar * cst.c**3) / (omega_0 * A)
        ) / (
            omega_0 * cst.e
        )  # <- anomalous constants I needed to add... hmm
        hyperfine_term = np.sqrt((2 * F2 + 1) * (2 * F1 + 1)) * wigner_6j(
            J1, J2, 2, F2, F1, I
        )

        # q -> polarization
        polarization_map = {
            -2: 1 / np.sqrt(6) * np.array([[1, 1j, 0], [1j, -1, 0], [0, 0, 0]]),
            -1: 1 / np.sqrt(6) * np.array([[0, 0, 1], [0, 0, 1j], [1, 1j, 0]]),
            0: 1 / 3 * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]]),
            1: 1 / np.sqrt(6) * np.array([[0, 0, -1], [0, 0, 1j], [-1, 1j, 0]]),
            2: 1 / np.sqrt(6) * np.array([[1, -1j, 0], [-1j, -1, 0], [0, 0, 0]]),
        }

        geometry_term = (
            np.sqrt(2 * J2 + 1)
            * wavevector.dot(np.matmul(polarization_map[q], polarization))
            * wigner_3j(F2, 2, F1, M2, -q, -M1)
        )

        return float(
            (abs(units_term) * abs(geometry_term) * abs(hyperfine_term)).evalf()
        )

    else:
        raise ValueError(
            "Currently only support dipole and quadrupole allowed transitions"
        )


########################################################################################


def rabi_from_intensity(laser, transition, intensity):
    """Method computing a transition's resonant rabi frequency addressed by a laser and its intensity

    Args:
        laser_index (int): 0-indexed integer pointing to the laser user would like set wavelength for.
                            Order based on chamber instantiation.
        intensity (float): laser intensity in W/m^2
        transition (Transition): transition object
    Returns:
        rabi_frequency (float)
    """

    matrix_elem = compute_matrix_element(laser, transition)
    E = (2 * intensity / (cst.epsilon_0 * cst.c)) ** 0.5

    return matrix_elem * E * cst.e / cst.hbar


def intensity(laser):
    matrix_elem = compute_matrix_element(laser, laser.transition)

    I = cst.c * cst.epsilon_0 / 2 * (cst.hbar * laser.rabi / (matrix_elem * cst.e)) ** 2
    return I
