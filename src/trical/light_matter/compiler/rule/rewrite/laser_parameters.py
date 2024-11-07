import numpy as np
import oqd_compiler_infrastructure as ci
from trical.light_matter.compiler.rule.rewrite.get_rabi_frequencies import ComputeMatrixElement
from typing import Callable
from oqd_compiler_infrastructure import RewriteRule
from trical.light_matter.utilities import D_mn
from qutip import Qobj, QobjEvo, basis
# from trical.backend.qutip.passes import QuantumOperator
from trical.misc import constants as cst

class SetState(ci.ConversionRule):

    """Rule for accessing a mode's nth excited state."""

    def apply(self, mode, operands):
        
        mode = operands['mode']
        n = operands['n']

        if n >= mode.N:
            raise ValueError("Outside of Hilbert space")
        return basis(mode.N, n)
    
class GroundState(ci.ConversionRule):
    
    """Rule for accessing a mode's ground state."""

    def apply(self, mode, operands):
        mode = operands['mode']
        return basis(mode.N, 0)
  
class ModeCutoff:
    @staticmethod
    def set_mode_cutoff(mode, val):
        mode.cutoff = val + 1


class SetLaserWavelengthFromTransition(ci.ConversionRule):
    def __init__(self, laser_index, transition):
        self.laser_index = laser_index
        self.transition = transition

    def apply(self, model):
        """Sets the laser wavelength to the transition wavelength."""
        lvl1, lvl2 = self.transition.level1, self.transition.level2
        f1, f2 = lvl1.energy, lvl2.energy

        wvl = cst.c / abs(f2 - f1)
        return wvl


class SetLaserIntensityFromPiTime(ci.ConversionRule):
    def __init__(self, laser_index, pi_time, transition):
        self.laser_index = laser_index
        self.pi_time = pi_time
        self.transition = transition

    def apply(self, model):
        laser = model.lasers[self.laser_index]

        compute_matrix_element_rule = ComputeMatrixElement()
        matrix_elem = compute_matrix_element_rule.compute_matrix_element(
            model=model, 
            operands={"laser": laser, "transition": self.transition}
        )

        I = (
            cst.eps_0
            * cst.c
            / 2
            * ((cst.hbar * np.pi) / (self.pi_time * matrix_elem * cst.e)) ** 2
        )
        
        return I


class SetLaserIntensityFromRabiFrequency(ci.ConversionRule):
    def apply(self, model, laser_index, rabi_frequency, transition):
        """Sets the laser intensity such that the resonant Rabi frequency is the one specified."""
        pi_time = np.pi / (2 * np.pi * rabi_frequency)

        set_pi_rule = SetLaserIntensityFromPiTime(laser_index, pi_time, transition)
        intensity_fn = set_pi_rule.apply(model)

        return intensity_fn
    
    
class ZeemanShift(RewriteRule):
    """Method that computes the Zeeman shift for the given level.
    Args:
        level (Level): Level object used for accessing quantum numbers.

    Returns:
        float: Zeeman shift in energy of the level in the presence of B.
    """

    def __init__(self, B):
        self.B = B

    def map_level(self, level):
        L = level.orbital
        J = level.spin_orbital
        F = level.spin_orbital_nuclear
        m_F = level.spin_orbital_nuclear_magnetization

        I = abs(F - J)
        S = abs(J - L)

        # Bohr magneton
        mu_B = 9.27400994e-24
        g_j = 3 / 2 + (S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))
        g_f = (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1)) * g_j

        return g_f * mu_B * self.B * m_F
    
# class NumberOperator(RewriteRule):
#     """Rewrite rule for constructing the full Hilbert space number operator for a given mode.
#         Args:
#             model: The input model (not used in this rule).

#         Returns:
#             QuantumOperator: The number operator in the full Hilbert space.
#     """

#     def __init__(self, ions, selected_modes, mode_indx, name):
#         super().__init__()
#         self.ions = ions
#         self.selected_modes = selected_modes
#         self.mode_indx = mode_indx
#         self.name = name

#     def apply(self, model):
#         ion_buffer = [qeye(ion.N_levels) for ion in self.ions]
#         mot_buffer = [qeye(mode.N) for mode in self.selected_modes]

#         dims = self.selected_modes[self.mode_indx].N
#         mot_buffer[self.mode_indx] = destroy(N=dims).dag() * destroy(N=dims)

#         return QuantumOperator(qobj=tensor(*ion_buffer, *mot_buffer), name=self.name)

