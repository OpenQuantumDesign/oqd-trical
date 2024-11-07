import numpy as np
from cmath import exp
from qutip import Qobj
import oqd_compiler_infrastructure as ci
from trical.misc import constants as cst
from trical.light_matter.interface.coefficient import *
from trical.light_matter.interface.operator import *
from trical.light_matter.interface.Chamber import *
from trical.light_matter.compiler.rule.rewrite.get_rabi_frequencies import *


class ConstructHTreeRule(ci.ConversionRule):
    def __init__(self, ions, modes, lasers, timescale, chamber):
        super().__init__()
        self.ions = ions
        self.modes = modes
        self.lasers = lasers
        self.timescale = timescale
        self.laser_ops = []
        self.chamber = chamber
        self.N_cutoff = 10

    def map_Chamber(self, model, operands):

        # After traversal, combine the collected operators
        chain_H = operands["chain"]
        total_laser_op = sum(self.laser_ops, start=Zero())
        full_H = chain_H + total_laser_op
        h_tree = full_H
        return h_tree

    def map_Chain(self, model, operands):
        dims_list = [len(ion.levels) for ion in model.ions]
        N = len(operands["ions"])
        L = len(operands["modes"])

        # # Map ions and store the operators
        # ion_op_list = []
        # for ion in operands["ions"]:
        #     op = ion
        #     ion_op_list.append(op)

        ion_ops = []
        for n in range(N):
            ops = [Identity(dims=dims_list[k]) if n != k else operands["ions"][n] for k in range(N)]
            ops.extend([Identity(dims=self.N_cutoff) for _ in range(L)])
            ion_ops.append(extended_kron(ops))

        # # Map modes and store the operators
        # mode_op_list = []
        # for mode in modes:
        #     op = self.map_VibrationalMode(mode)
        #     mode_op_list.append(op)

        mode_ops = []
        for l in range(L):
            ops = [Identity(dims=dims_list[n]) for n in range(N)]
            ops.extend([Identity(dims=self.N_cutoff) if l != k else operands["modes"][l] for k in range(L)])
            mode_ops.append(extended_kron(ops))

        # Store the chain operators for later combination
        chain_ops = ion_ops + mode_ops

        return sum(chain_ops, start=Zero())

    def map_Ion(self, model, operands):
        levels = model.levels
        J = len(levels)
        op = Zero()
        for j, (label, level) in enumerate(levels.items()):
            energy = level.energy * self.timescale
            coeff = ConstantCoefficient(value=energy)
            ketbra = KetBra(i=j, j=j, dims=J)
            op += OperatorScalarMul(op=ketbra, coeff=coeff)
        return op

    def map_VibrationalMode(self, model, operands):
        omega = 2 * np.pi * model.eigenfreq * self.timescale
        coeff = ConstantCoefficient(value=omega)
        creation = Creation(dims=self.N_cutoff)
        annihilation = Annihilation(dims=self.N_cutoff)
        mode_H = OperatorScalarMul(op=OperatorMul(op1=creation, op2=annihilation), coeff=coeff)
        return mode_H


    def map_Laser(self, model, operands):
        laser = model
        ions = self.ions
        modes = self.modes
        lasers = self.lasers
        timescale = self.timescale

        laser_op = Zero()
        rabi_rule = RabiFrequencyFromIntensity()

        for n, ion in enumerate(ions):
            levels = ion.levels
            level_labels = list(levels.keys())
            J_n = len(levels)

            for transition in ion.transitions:
                lbl1, lbl2 = transition
                lvl1 = ion.transitions[transition].level1
                lvl2 = ion.transitions[transition].level2

                I = laser.intensity 

                Omega_nm = rabi_rule.compute_rabi_frequency_from_intensity(
                    ion,
                    {
                        'laser_index': lasers.index(laser), 
                        'lasers': lasers, 
                        'intensity': I,
                        'transition': ion.transitions[transition],
                        'chamber': self.chamber
                    },
                ) * timescale

                f = abs(lvl2.energy - lvl1.energy)
                omega_0 = 2 * np.pi * f * timescale
                laser_freq = (
                    2 * np.pi * (cst.c / laser.wavelength + laser.detuning) * timescale
                )
                Delta_nmkj = laser_freq - omega_0

                int_coup_int_buffer = [
                    Identity(dims=len(ion_int.levels)) for ion_int in ions
                ]
                idx_lbl2 = level_labels.index(lbl2)
                idx_lbl1 = level_labels.index(lbl1)
                int_coup_int_buffer[n] = KetBra(i=idx_lbl2, j=idx_lbl1, dims=J_n)

                int_coup_mot_buffer = [
                    Identity(dims=self.N_cutoff + 1) for _ in modes
                ]
                int_term = extended_kron(int_coup_int_buffer + int_coup_mot_buffer)

                mot_coup_int_term = [
                    Identity(dims=len(ion_int.levels)) for ion_int in ions
                ]
                mot_coup_mot_term = [
                    Identity(dims=self.N_cutoff + 1) for _ in modes
                ]

                mot_term = extended_kron(mot_coup_int_term + mot_coup_mot_term)

                # Rotating term: e^{i(k·r - ωt + phi)}
                rabi_prefactor_rot = WaveCoefficient(
                    amplitude=Omega_nm / 2,
                    frequency=-Delta_nmkj,
                    phase=laser.phase,
                    ion_indx=n,
                    laser_indx=self.lasers.index(laser),  
                    i=idx_lbl2,
                    j=idx_lbl1,
                    mode_indx=None,
                )

                # Create the operator for the rotating term
                inner_ion_H_rot = OperatorMul(
                    op1=OperatorScalarMul(coeff=rabi_prefactor_rot, op=int_term),
                    op2=mot_term,
                )

                # Add the rotating term and its Hermitian conjugate
                laser_op += inner_ion_H_rot + hc(op=inner_ion_H_rot)

        self.laser_ops.append(laser_op)

        return None


    # def hc(self, operator):
    #     return hc(op=operator)

                    # **Removed Lamb-Dicke approximation and related code**
                # For now, we'll set the motional coupling terms to Identity,
                # effectively decoupling the motional modes from the laser interaction.
                #
                # for l, mode in enumerate(modes):
                #     nu = 2 * np.pi * mode.eigenfreq * timescale
                #     eta_nml = lambdicke(mode, ion, laser)
                #     alpha = WaveCoefficient(
                #         amplitude=eta_nml,
                #         frequency=nu,
                #         phase=np.pi / 2,
                #         ion_indx=None,
                #         laser_indx=None,
                #         i=None,
                #         j=None,
                #         mode_indx=l
                #     )
                #     mot_coup_mot_term[l] = Displacement(alpha=alpha, dims=self.mode_cutoff_value + 1)











# class ConstructHTreeRule(ci.ConversionRule):
#     def __init__(self):
#         super().__init__()
#         self.full_H = Zero() 

#     def map_Chamber(self, mode, operands):
#         """Processes the chamber component."""
#         pass

#     def map_Chain(self, mode, operands):
#         """Construct operators for ions and modes in the Chain."""
#         ions = operands['ions']
#         modes = operands['modes']
#         N = len(ions)
#         L = len(modes)

#         op = []
#         for n, ion in enumerate(ions):
#             ion_op = [Identity() if n != k else ion for k in range(N)]
#             ion_op.extend([Identity() for _ in range(L)])
#             op.append(self.extended_kron(ion_op))

#         for l, mode in enumerate(modes):
#             mode_op = [Identity() for _ in range(N)]
#             mode_op.extend([Identity() if l != k else mode for k in range(L)])
#             op.append(self.extended_kron(mode_op))

#         return op 

#     def map_Ion(self, ion, operands):
#         """Construct operator for the energy levels of the Ion."""
#         J = len(ion.levels)
#         op = Zero()

#         for j, level in enumerate(ion.levels.values()):
#             op += KetBra(i=j, j=j, dims=J) * ConstantCoefficient(value=level.energy)

#         return op

#     def map_VibrationalMode(self, mode, operands):
#         """Construct the operator for the Vibrational Mode."""
#         return Creation() * Annihilation() * ConstantCoefficient(value=mode.eigenfreq)

#     def map_Laser(self, mode, operands):
#         """Process laser interaction with ions and modes."""
#         lasers = operands['lasers']
#         timescale = operands['timescale']
#         ions = mode.chain.ions
#         modes = mode.chain.modes
#         # Iterate over lasers and ions to process interactions
#         for m, laser in enumerate(lasers):
#             for n, ion in enumerate(ions):
#                 self.process_ion_and_laser_interaction(ion, laser, m, n, ions, modes, timescale)
#         if self.full_H is None:
#             self.full_H = Zero()

#         return self.full_H + hc(self.full_H)  # Ensuring Hermitian conjugate is applied

#     def extended_kron(self, op_list):
#         """Recursive function for tensor product of operators."""
#         if len(op_list) == 2:
#             return OperatorKron(op1=op_list[0], op2=op_list[1])
#         return OperatorKron(op1=op_list[0], op2=self.extended_kron(op_list[1:]))

#     def process_ion_and_laser_interaction(self, ion, laser, m, n, ions, modes, timescale):
#         """Handles the interaction between a specific ion and laser."""
#         for transition in ion.transitions:
#             lbl1, lbl2 = transition
#             lvl1 = ion.transitions[transition].level1
#             lvl2 = ion.transitions[transition].level2

#             I = laser.I(0)
#             Omega_nm = (
#                 ion.rabi_frequency_from_intensity(
#                     laser_index=m, intensity=I, transition=ion.transitions[transition]
#                 ) * timescale
#             )

#             f = abs(lvl2.energy - lvl1.energy)
#             omega_0 = 2 * np.pi * f * timescale
#             laser_freq = 2 * np.pi * (cst.c / laser.wavelength + laser.detuning) * timescale
#             Delta_nmkj = laser_freq - omega_0

#             int_term = self.construct_int_term(ion, ions, lbl1, lbl2, modes, n)
#             mot_term = self.construct_mot_term(ion, laser, modes, n, timescale)

#             rabi_prefactor = WaveCoefficient(
#                 amplitude=Omega_nm / 2,
#                 frequency=-Delta_nmkj,
#                 phase=laser.phi,
#                 ion_indx=n,
#                 laser_indx=m,
#                 i=list(ion.state.keys()).index(lbl2),
#                 j=list(ion.state.keys()).index(lbl1),
#                 mode_indx=None,
#             )

#             inner_ion_H = OperatorMul(
#                 op1=OperatorScalarMul(coeff=rabi_prefactor, op=int_term),
#                 op2=mot_term,
#             )

#             self.full_H += inner_ion_H  # Accumulate the interaction term

#     def construct_int_term(self, ion, ions, lbl1, lbl2, modes, n):
#         """Constructs the interaction term for internal states."""
#         J_n = ion.N_levels
#         int_coup_int_buffer = [Identity(dims=ion_int.N_levels) for ion_int in ions]
#         int_coup_int_buffer[n] = KetBra(
#             i=list(ion.state.keys()).index(lbl2),
#             j=list(ion.state.keys()).index(lbl1),
#             dims=J_n,
#         )
#         int_coup_mot_buffer = [Identity(dims=mode.N) for mode in modes]
#         return self.extended_kron(int_coup_int_buffer + int_coup_mot_buffer)

#     def construct_mot_term(self, ion, ions, laser, modes, n, timescale):
#         """Constructs the motional coupling term."""
#         mot_coup_int_term = [Identity(dims=ion_int.N_levels) for ion_int in ions]
#         mot_coup_mot_term = [Identity(dims=mode.N) for mode in modes]

#         for l, mode in enumerate(modes):
#             nu = 2 * np.pi * mode.eigenfreq * timescale
#             eta_nml = lambdicke(mode, ion, laser)
#             alpha = WaveCoefficient(
#                 amplitude=eta_nml,
#                 frequency=nu,
#                 phase=np.pi / 2,
#                 ion_indx=None,
#                 laser_indx=None,
#                 i=None,
#                 j=None,
#                 mode_indx=l,
#             )
#             mot_coup_mot_term[l] = Displacement(alpha=alpha, dims=mode.N)

#         return self.extended_kron(mot_coup_int_term + mot_coup_mot_term)

