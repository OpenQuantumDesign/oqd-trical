import oqd_compiler_infrastructure as ci
import numpy as np
from typing import Dict, Tuple
from trical.misc import constants as cst
from sympy.physics.wigner import wigner_3j, wigner_6j
from functools import lru_cache 
from oqd_compiler_infrastructure import RewriteRule
from trical.light_matter.interface.Chamber import *

from trical.light_matter.interface.operator import (
    OperatorKron,
    WaveCoefficient,
)

class GetRabiFrequenciesDetunings(RewriteRule):
    """Rewrite rule class for traversing the Hamiltonian tree and extracting the Rabi frequencies and detunings

    Attributes:
        rabis (Dict[Tuple[int,int,int,int],float]): Dictionary mapping (ion_indx, laser_indx, i, j) -> rabi; acquired from GetRabiFrequencyDetunings's traversal
        detunings (Dict[Tuple[int,int,int,int],float]): Dictionary mapping (ion_indx, laser_indx, i, j) -> detuning; acquired from GetRabiFrequencyDetunings's traversal
    """

    def __init__(self):
        super().__init__()

        self.rabis = {}
        self.detunings = {}

    def map_OperatorScalarMul(self, model):
        """
        Args:
            model (OperatorScalarMul): traverses tree, looking for instances of OperatorScalarMul

        Returns:
            model (OperatorScalarMul): does not change the tree; instead, makes rabi and detuning dictionary accessible for external access
        """

        coeff = model.coeff
        op = model.op

        if isinstance(op, OperatorKron) and isinstance(coeff, WaveCoefficient):
            rabi = 2 * coeff.amplitude

            # The signed detuning info is stored in the frequency argument
            Delta = coeff.frequency

            # Index info
            i, j = coeff.i, coeff.j
            ion_indx, laser_indx = coeff.ion_indx, coeff.laser_indx

            self.rabis[(ion_indx, laser_indx, i, j)] = rabi
            self.detunings[(ion_indx, laser_indx, i, j)] = Delta

        return model
    
class RabiFrequencyFromIntensity(ci.ConversionRule):
    """Rule for computing the Rabi frequency from laser intensity and transition."""
    
    def compute_rabi_frequency_from_intensity(self, model, operands) -> float:        
        laser_index = operands['laser_index']
        lasers = operands['lasers']
        laser = lasers[laser_index]
        intensity = operands['intensity']
        transition = operands['transition']
        chamber = operands['chamber']

        laser = lasers[laser_index]
        compute_matrix_element_rule = ComputeMatrixElement()
        matrix_elem = compute_matrix_element_rule.compute_matrix_element(
            model,
            {
                'laser': laser,
                'transition': transition,
                'chamber': chamber,
            }
        )        
        E = np.sqrt(2 * intensity / (cst.eps_0 * cst.c))

        return matrix_elem * E * cst.e / cst.hbar


class ComputeMatrixElement(ci.ConversionRule):
    def __init__(self):
        super().__init__()
        # polarization maps
        self.polarization_map_E1 = {
            -1: (1 / np.sqrt(2)) * np.array([1, 1j, 0]),
            0: np.array([0, 0, 1]),
            1: (1 / np.sqrt(2)) * np.array([1, -1j, 0]),
        }
        self.polarization_map_E2 = {
            -2: (1 / np.sqrt(6)) * np.array([[1, 1j, 0], [1j, -1, 0], [0, 0, 0]]),
            -1: (1 / np.sqrt(6)) * np.array([[0, 0, 1], [0, 0, 1j], [1, 1j, 0]]),
            0: (1 / 3) * np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 2]]),
            1: (1 / np.sqrt(6)) * np.array([[0, 0, -1], [0, 0, 1j], [-1, 1j, 0]]),
            2: (1 / np.sqrt(6)) * np.array([[1, -1j, 0], [-1j, -1, 0], [0, 0, 0]]),
        }

    def compute_matrix_element(self, model, operands) -> float:
        laser = operands['laser']
        transition = operands['transition']
        chamber = operands['chamber']

        lvl1, lvl2 = transition.level1, transition.level2
        J1, J2 = lvl1.spin_orbital, lvl2.spin_orbital
        F1, F2 = lvl1.spin_orbital_nuclear, lvl2.spin_orbital_nuclear

        M1 = lvl1.spin_orbital_nuclear_magnetization if lvl1.spin_orbital_nuclear_magnetization is not None else 0.0
        M2 = lvl2.spin_orbital_nuclear_magnetization if lvl2.spin_orbital_nuclear_magnetization is not None else 0.0

        q = M2 - M1

        polarization = np.asarray(laser.polarization)
        wavevector = np.asarray(laser.wavevector)
        Bhat = np.asarray(chamber.Bhat)

        if not np.allclose(Bhat, np.array([0, 0, 1])):
            polarization, wavevector = self.rotate_vectors(polarization, wavevector, Bhat)

        E1, E2 = lvl1.energy, lvl2.energy

        if E1 > E2:
            raise ValueError("Expected E2 > E1")
        if lvl1.nuclear != lvl2.nuclear:
            raise ValueError("Different nuclear spins between two levels in transition:", transition)

        I = lvl1.nuclear
        A = transition.einsteinA
        omega_0 = 2 * np.pi * (E2 - E1)

        common_prefactor = np.sqrt((2 * F2 + 1) * (2 * F1 + 1))
        
        if transition.multipole == 'E1':
            multipole_value = 1  
        elif transition.multipole == 'E2':
            multipole_value = 2 
        else:
            raise ValueError(f"Invalid multipole type: {transition.multipole}")

        wigner_3j_value = float(wigner_3j(F2, multipole_value, F1, M2, -q, -M1))
        wigner_6j_value = float(wigner_6j(J1, J2, multipole_value, F2, F1, I))

        if transition.multipole == "E1":
            units_term = np.sqrt((3 * np.pi * cst.eps_0 * cst.hbar * cst.c ** 3) / (omega_0 ** 3) * A) / cst.e
            hyperfine_term = common_prefactor * wigner_6j_value

            polarization_vector = self.polarization_map_E1.get(q)
            if polarization_vector is None:
                raise ValueError(f"Invalid value for q: {q} in E1 transition.")

            geometry_term = (
                np.sqrt(2 * J2 + 1)
                * np.dot(polarization_vector, polarization)
                * wigner_3j_value
            )

        elif transition.multipole == "E2":
            units_term = np.sqrt((15 * np.pi * cst.eps_0 * cst.hbar * cst.c ** 3) / (omega_0 ** 5) * A) / cst.e
            hyperfine_term = common_prefactor * wigner_6j_value

            polarization_matrix = self.polarization_map_E2.get(q)
            if polarization_matrix is None:
                raise ValueError(f"Invalid value for q: {q} in E2 transition.")

            geometry_term = (
                np.sqrt(2 * J2 + 1)
                * np.dot(wavevector, np.matmul(polarization_matrix, polarization))
                * wigner_3j_value
            )

        else:
            raise ValueError("Currently only support dipole and quadrupole allowed transitions")

        total_term = units_term * geometry_term * hyperfine_term
        return abs(total_term)

    def rotate_vectors(self, polarization, wavevector, Bhat):
        """Rotates polarization and wavevector vectors if the B-field is not aligned with the z-axis."""
        
        a = np.cross(Bhat, np.array([0, 0, 1]))
        norm_a = np.linalg.norm(a)
        
        if norm_a == 0:
            return polarization, wavevector 
        
        a /= norm_a
        
        theta = np.arccos(Bhat[2])
        sin_theta = np.sin(theta)
        sin_theta_half_squared = np.sin(theta / 2) ** 2
        
        amatrix = np.array([
            [0, -a[2], a[1]],
            [a[2], 0, -a[0]],
            [-a[1], a[0], 0]
        ])
        
        R = (
            np.identity(3)
            + sin_theta * amatrix
            + 2 * sin_theta_half_squared * np.dot(amatrix, amatrix)
        )
        
        polarization_rot = np.dot(R, polarization)
        wavevector_rot = np.dot(R, wavevector)
        
        return polarization_rot, wavevector_rot
