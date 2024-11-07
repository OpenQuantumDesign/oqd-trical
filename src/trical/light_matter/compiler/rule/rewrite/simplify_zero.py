import numpy as np
from typing import Dict, Tuple
from math import cos, sin, sqrt, atan2
from functools import reduce
from trical.light_matter.compiler.rule.approximation import RWA_and_LD_Approximations
from oqd_compiler_infrastructure import (
    Post,
    RewriteRule,
    extract_alphas,
    Identity,
    Displacement,
    KetBra,
    OperatorMul,
)
from ....interface import (
    OperatorKron,
    WaveCoefficient,
    Zero,
    OperatorScalarMul,
)

# FUNCTIONS BELOW CLASSES NEED FIXING CIRCULAR IMPORTS WITH ADIABATIC ELIM: SUBSTITUTION AND RWA_LD_APPROX

class SimplifyZero(RewriteRule):
    """Rewrite rule for simplifying Zero objects introduced when eliminating coupling terms."""

    def map_OperatorAdd(self, model):
        """Simplify OperatorAdd when involving Zero objects."""
        op1, op2 = model.op1, model.op2

        if isinstance(op1, Zero):
            return op2 if not isinstance(op2, Zero) else Zero()
        return op1 if isinstance(op2, Zero) else model
    
class ReorderScalarMul(RewriteRule):
    """ReWrite rule for reordering terms in the Hamiltonian tree to facilitate taking the LD, RWA approximations"""

    def map_OperatorMul(self, model):
        """Method for moving the location of the WaveCoefficient prefactor

        Args:
            model (OperatorMul): OperatorMul object
        """

        op1 = model.op1
        op2 = model.op2

        if isinstance(op1, OperatorScalarMul) and isinstance(op2, OperatorKron):

            if isinstance(op1.op, OperatorKron) and isinstance(
                op1.coeff, WaveCoefficient
            ):
                coeff = op1.coeff
                int_term = op1.op
                mot_term = op2

                return OperatorMul(
                    op1=int_term, op2=OperatorScalarMul(coeff=coeff, op=mot_term)
                )

        else:
            return model

def approximate(tree, n_cutoff, timescale, ld_cond_th=1e-2, rwa_cutoff="inf"):
    """Master function for performing both the Lamb-Dicke and rotating wave approximations (RWA)

    Args:
        tree (Operator): Hamiltonian tree whose terms are joined via OperatorAdd's
        n_cutoff (int): max phonon number for a given mode
        timescale (float): time unit (e.g. 1e-6 for microseconds)
        ld_cond_th (float): threshold on Lamb-Dicke approximation conditions
        rwa_cutoff (Union[float,str]): all terms rotating faster than rwa_cutoff are set to 0. Acceptable str is 'inf'

    Returns:
        approx_tree (Operator):  Hamiltonian tree post LD and RWA approximations

    """
    reorder = Post(ReorderScalarMul())

    reordered_tree = reorder(tree)

    approximator = Post(
        RWA_and_LD_Approximations(
            n_cutoff=n_cutoff,
            rwa_cutoff=rwa_cutoff,
            timescale=timescale,
            ld_cond_th=ld_cond_th,
        )
    )

    approx_tree = approximator(reordered_tree)
    return approx_tree

def extended_kron(op_list):
    """Create an AST representation of the tensor product of many operators."""
    return reduce(lambda x, y: OperatorKron(op1=x, op2=y), op_list)

def simplify_complex_subtraction(wc1, wc2):
    """Simplify the difference between two WaveCoefficient objects with the same frequency."""
    if wc1.frequency != wc2.frequency:
        raise ValueError("Helper function expects both frequencies to be the same")

    # Convert to rectangular form
    re1 = wc1.amplitude * cos(wc1.phase)
    im1 = wc1.amplitude * sin(wc1.phase)
    re2 = wc2.amplitude * cos(wc2.phase)
    im2 = wc2.amplitude * sin(wc2.phase)

    # Subtract and convert back to polar form
    re, im = re1 - re2, im1 - im2
    amplitude = sqrt(re**2 + im**2)
    phase = atan2(im, re)

    return WaveCoefficient(
        amplitude=amplitude,
        frequency=wc1.frequency,
        phase=phase,
        ion_indx=None,
        laser_indx=None,
        mode_indx=None,
        i=None,
        j=None,
    )

def substitute(
    chopped_tree,
    old_info,
    two_level_info,
    N,
    L,
    ion_N_levels,
    mode_cutoffs,
    detunings,
    transformations,
):
    """Insert effective terms into the Hamiltonian tree after two-photon adiabatic elimination."""
    print("Transformations:")
    for ion_indx, trans_dict in transformations.items():
        for key, values in trans_dict.items():
            print(f"{values[0]} -> {key} in ion {ion_indx}.")

    new_tree = chopped_tree

    for ion_indx, ion_info in two_level_info.items():
        mot_terms_n = old_info["mot_terms"][ion_indx]
        J_n = ion_N_levels[ion_indx]

        for (i, j), level_info in ion_info.items():
            rabi_eff = level_info["rabi_eff"]
            laser_indices = level_info["lasers"]
            phi_list = [old_info["lasers"][idx]["phase"] for idx in laser_indices]
            phi_eff = phi_list[0] - sum(phi_list[1:])

            int_coup_int_buffer = [Identity(dims=ion_N_levels[n]) for n in range(N)]
            int_coup_int_buffer[ion_indx] = KetBra(i=i, j=j, dims=J_n)
            int_coup_mot_buffer = [Identity(dims=mode_cutoffs[l]) for l in range(L)]
            new_int_term = extended_kron(int_coup_int_buffer + int_coup_mot_buffer)

            eliminated_pairs = transformations[ion_indx][(i, j)]
            alphas = {
                idx: extract_alphas(mot_terms_n[idx][eliminated_pairs[0]])
                for idx in laser_indices
            }
            alphas_list = [alphas[idx] for idx in laser_indices]
            alphas_eff = alphas_list[0][:]

            for laser_alphas in alphas_list[1:]:
                alphas_eff = [
                    simplify_complex_subtraction(a_eff, a)
                    for a_eff, a in zip(alphas_eff, laser_alphas)
                ]

            mot_coup_int_buffer = [Identity(dims=ion_N_levels[n]) for n in range(N)]
            mot_coup_mot_buffer = [Identity(dims=mode_cutoffs[l]) for l in range(L)]

            for a, alpha_eff in enumerate(alphas_eff):
                mot_coup_mot_buffer[a] = Displacement(
                    alpha=alpha_eff, dims=mode_cutoffs[a]
                )

            new_mot_term = extended_kron(mot_coup_int_buffer + mot_coup_mot_buffer)
            delta_eff = detunings[(ion_indx, 0, 2, 0)] - detunings[(ion_indx, 1, 2, 1)]

            rabi_prefactor = WaveCoefficient(
                amplitude=rabi_eff / 2,
                frequency=delta_eff,
                phase=phi_eff,
                ion_indx=None,
                laser_indx=None,
                mode_indx=None,
                i=None,
                j=None,
            )

            new_two_level_tree = OperatorMul(
                op1=OperatorScalarMul(coeff=rabi_prefactor, op=new_int_term),
                op2=new_mot_term,
            )
            new_tree += new_two_level_tree

    return new_tree
