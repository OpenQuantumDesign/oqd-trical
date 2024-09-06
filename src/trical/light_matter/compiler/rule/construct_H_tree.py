from cmath import exp

import numpy as np

########################################################################################

from ....misc import constants as cst
from ...interface import (
    WaveCoefficient,
    OperatorScalarMul,
    KetBra,
    Displacement,
    Identity,
    OperatorKron,
    hc,
    OperatorMul,
)
from ...compiler import Hamiltonian
from ...utilities import lambdicke

########################################################################################


def extended_kron(op_list):
    """Recursive convenience function for creating AST representation of the tensor product of many operators

    Args:
        op_list (list): List of Operator-like objects as defined in the AST

    Returns:
        extended_kron (OperatorKron): Nested tensor product of operators in the operator list
    """
    if len(op_list) == 2:
        return OperatorKron(op1=op_list[0], op2=op_list[1])

    return extended_kron([op_list[0], extended_kron(op_list[1:])])


def construct_H_tree(chamber, timescale):
    """
    Constructs system Hamiltonian tree in the interaction picture using the abstact syntax language
    defined in analog_Hamiltonian_AST.py. For more information about how this Hamiltonian was derived
    please visit the documentation supplement here:

    https://www.overleaf.com/read/dptrvjrtbjbf#4b0ace

    Args:
        chamber (Chamber): Chamber object as defined in Chamber.py
        timescale (float): float representing time unit; e.g. 1e-6 for micro-seconds

    Returns:
        hamiltonian (Hamiltonian): Hamiltonian object as defined in hamiltonian.py
    """

    chain = chamber.chain
    lasers = chamber.lasers
    ions = chamber.chain.ions
    modes = chain.selected_modes

    # Frequencies should be multiplied by timescale; times should be divided by timescale

    full_H = None

    for m, laser in enumerate(lasers):
        for n, ion in enumerate(ions):

            J_n = ion.N_levels

            for transition in ion.transitions:

                (
                    lbl1,
                    lbl2,
                ) = transition

                lvl1, lvl2 = (
                    ion.transitions[transition].level1,
                    ion.transitions[transition].level2,
                )

                I = laser.I(
                    0
                )  # intensity generally some function of time to be evaluated at each time step

                Omega_nm = (
                    chamber.rabi_frequency_from_intensity(
                        laser_index=m,
                        intensity=I,
                        transition=ion.transitions[transition],
                    )
                    * timescale
                )

                # Construct first term in Hamiltonian

                # Already accounted for Zeeman shift when constructing chamber

                f = abs(lvl2.energy - lvl1.energy)

                omega_0 = 2 * np.pi * f * timescale  # transition frequency

                laser_freq = (
                    2 * np.pi * (cst.c / laser.wavelength + laser.detuning) * timescale
                )

                Delta_nmkj = (
                    laser_freq - omega_0
                )  # no need to multiply by timescale since laser_freq and omega_0 already scaled!

                # Atomic transition part:

                # 'buffers' are lists of operators representing some part of the Hilbert space
                # e.g. int_coup_int_buffer contains the term addressing the ion (internal) Hilbert space
                int_coup_int_buffer = [
                    Identity(dims=ion_int.N_levels) for ion_int in ions
                ]
                int_coup_int_buffer[n] = KetBra(
                    i=list(ion.state.keys()).index(lbl2),
                    j=list(ion.state.keys()).index(lbl1),
                    dims=J_n,
                )

                int_coup_mot_buffer = [Identity(dims=mode.N) for mode in modes]

                # Full Hilbert space Pauli plus

                int_term = extended_kron(int_coup_int_buffer + int_coup_mot_buffer)

                # Motional mode part:

                mot_coup_int_term = [
                    Identity(dims=ion_int.N_levels) for ion_int in ions
                ]
                mot_coup_mot_term = [Identity(dims=mode.N) for mode in modes]

                for l, mode in enumerate(modes):

                    nu = 2 * np.pi * mode.eigenfreq * timescale

                    eta_nml = lambdicke(mode, ion, laser)

                    # put i in alpha as part of phase: pi/2, instead of amplitude
                    alpha = WaveCoefficient(
                        amplitude=eta_nml,
                        frequency=nu,
                        phase=np.pi / 2,
                        ion_indx=None,
                        laser_indx=None,
                        i=None,
                        j=None,
                        mode_indx=l,
                    )

                    mot_coup_mot_term[l] = Displacement(alpha=alpha, dims=mode.N)

                mot_term = extended_kron(mot_coup_int_term + mot_coup_mot_term)

                rabi_prefactor = WaveCoefficient(
                    amplitude=Omega_nm / 2,
                    frequency=-Delta_nmkj,
                    phase=laser.phi,
                    ion_indx=n,
                    laser_indx=m,
                    i=list(ion.state.keys()).index(lbl2),
                    j=list(ion.state.keys()).index(lbl1),
                    mode_indx=None,
                )

                inner_ion_H = OperatorMul(
                    op1=OperatorScalarMul(coeff=rabi_prefactor, op=int_term),
                    op2=mot_term,
                )

                if full_H is None:
                    full_H = inner_ion_H
                else:
                    full_H += inner_ion_H

    """
    Instead of passing in all the ion and mode objects, just let the Hamiltonian keep track of
    how many levels each ion has and what the mode_cutoff is for each mode. This is useful information
    when constructing the effective Hamiltonian in Raman transitions.
    """

    ion_N_levels = {}
    for n, ion in enumerate(ions):
        ion_N_levels[n] = ion.N_levels

    mode_cutoffs = {}
    for l, mode in enumerate(modes):
        mode_cutoffs[n] = mode.N

    return Hamiltonian(
        tree=full_H + hc(),
        args={"chamber": chamber, "timescale": timescale},
        N=len(ions),
        M=len(lasers),
        L=len(modes),
        ion_N_levels=ion_N_levels,
        mode_cutoffs=mode_cutoffs,
    )
