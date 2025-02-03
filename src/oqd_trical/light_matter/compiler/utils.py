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

import numpy as np
from sympy.physics.wigner import wigner_3j, wigner_6j

########################################################################################
from oqd_trical.misc import constants as cst

########################################################################################


def compute_matrix_element(laser, transition):
    """Computes matrix element corresponding to a laser interacting with a particular transition

    Args:
        laser (Beam): laser to compute matrix element of
        transition (Transition): transition to compute matrix element of

    Returns:
        matrix_element (float): Multipole matrix elements corresponding to the interaction between the laser and the transition

    Warning:
        Currently implemented for only `E1` and `E2` transitions.
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

    omega = E2 - E1

    # TODO M1 transition multipole
    if transition.multipole == "M1":
        raise NotImplementedError

    if transition.multipole == "E1":
        units_term = np.sqrt(
            (3 * np.pi * cst.epsilon_0 * cst.hbar * cst.c**3 * A)
            / (omega**3 * cst.e**2)
        )
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
            (15 * np.pi * cst.epsilon_0 * cst.hbar * cst.c**3) / (omega * A)
        ) / (omega * cst.e)  # <- anomalous constants I needed to add... hmm
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
    """Computes a transition's resonant rabi frequency when addressed by a laser with intensity

    Args:
        laser (Beam): laser to compute resonant rabi frequency of
        transition (Transition): transition to compute resonant rabi frequency of
        intensity (float): intensity of laser

    Returns:
        rabi_frequency (float): resonant rabi frequency corresponding to the interaction between the laser and transition
    """

    matrix_elem = compute_matrix_element(laser, transition)
    E = (2 * intensity / (cst.epsilon_0 * cst.c)) ** 0.5

    return matrix_elem * E * cst.e / cst.hbar


def intensity_from_laser(laser):
    """Computes the intensity from a laser

    Args:
        laser (Beam): laser to compute intensity of.

    Returns:
        intensity (float): intensity of the laser
    """
    matrix_elem = compute_matrix_element(laser, laser.transition)

    I = cst.c * cst.epsilon_0 / 2 * (cst.hbar * laser.rabi / (matrix_elem * cst.e)) ** 2

    return I
