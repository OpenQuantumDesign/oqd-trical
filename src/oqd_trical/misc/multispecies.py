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
Module containing useful functions relavent for multi-species systems.
"""

########################################################################################

import numpy as np

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
