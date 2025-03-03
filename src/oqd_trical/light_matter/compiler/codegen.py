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

from functools import reduce

import numpy as np
from oqd_compiler_infrastructure import ConversionRule

from oqd_trical.light_matter.compiler.utils import (
    intensity_from_laser,
    rabi_from_intensity,
)
from oqd_trical.light_matter.interface.emulator import (
    AtomicEmulatorCircuit,
    AtomicEmulatorGate,
)
from oqd_trical.light_matter.interface.operator import (
    Annihilation,
    Creation,
    Displacement,
    Identity,
    KetBra,
    WaveCoefficient,
)
from oqd_trical.misc import constants as cst

########################################################################################


class ConstructHamiltonian(ConversionRule):
    """Maps an AtomicCircuit to an AtomicEmulatorCircuit replaces laser descriptions of operations with Hamiltonian description of operations"""

    def map_AtomicCircuit(self, model, operands):
        gates = (
            [operands["protocol"]]
            if isinstance(operands["protocol"], AtomicEmulatorGate)
            else operands["protocol"]
        )
        for gate in gates:
            gate.hamiltonian = gate.hamiltonian + operands["system"]

        return AtomicEmulatorCircuit(sequence=gates)

    def map_System(self, model, operands):
        self.N = len(model.ions)
        self.M = len(model.modes)

        self.ions = model.ions
        self.modes = model.modes

        ops = []
        for n, ion in enumerate(model.ions):
            ops.append(
                reduce(
                    lambda x, y: x @ y,
                    [
                        (
                            self._map_Ion(ion, n)
                            if i == n
                            else Identity(
                                subsystem=f"E{i}" if i < self.N else f"P{i - self.N}"
                            )
                        )
                        for i in range(self.N + self.M)
                    ],
                )
            )

        for m, mode in enumerate(model.modes):
            ops.append(
                reduce(
                    lambda x, y: x @ y,
                    [
                        (
                            self._map_Phonon(mode, m)
                            if i == self.N + m
                            else Identity(
                                subsystem=f"E{i}" if i < self.N else f"P{i - self.N}"
                            )
                        )
                        for i in range(self.N + self.M)
                    ],
                )
            )

        op = reduce(lambda x, y: x + y, ops)
        return op

    def _map_Ion(self, model, index):
        ops = [
            WaveCoefficient(amplitude=level.energy, frequency=0, phase=0)
            * KetBra(ket=n, bra=n, subsystem=f"E{index}")
            for n, level in enumerate(model.levels)
        ]

        op = reduce(lambda x, y: x + y, ops)
        return op

    def _map_Phonon(self, model, index):
        return WaveCoefficient(amplitude=model.energy, frequency=0, phase=0) * (
            Creation(subsystem=f"P{index}") * Annihilation(subsystem=f"P{index}")
        )

    def map_Beam(self, model, operands):
        I = intensity_from_laser(model)  # noqa: E741

        angular_frequency = (
            abs(model.transition.level2.energy - model.transition.level1.energy)
            + model.detuning
        )
        wavevector = angular_frequency * np.array(model.wavevector) / cst.c

        ops = []
        if self.modes:
            displacement_plus = []
            displacement_minus = []
            for m, mode in enumerate(self.modes):
                eta = np.dot(
                    wavevector,
                    mode.eigenvector[model.target * 3 : model.target * 3 + 3],
                ) * np.sqrt(
                    cst.hbar
                    / (2 * self.ions[model.target].mass * cst.m_u * mode.energy)
                )

                displacement_plus.append(
                    Displacement(
                        alpha=WaveCoefficient(
                            amplitude=eta, frequency=0, phase=np.pi / 2
                        ),
                        subsystem=f"P{m}",
                    )
                )
                displacement_minus.append(
                    Displacement(
                        alpha=WaveCoefficient(
                            amplitude=eta, frequency=0, phase=-np.pi / 2
                        ),
                        subsystem=f"P{m}",
                    )
                )

            displacement_plus = reduce(lambda x, y: x @ y, displacement_plus)
            displacement_minus = reduce(lambda x, y: x @ y, displacement_minus)

            for transition in self.ions[model.target].transitions:
                rabi = rabi_from_intensity(model, transition, I)

                ops.append(
                    (
                        reduce(
                            lambda x, y: x @ y,
                            [
                                (
                                    WaveCoefficient(
                                        amplitude=rabi / 2,
                                        frequency=-angular_frequency,
                                        phase=model.phase,
                                    )
                                    * (
                                        KetBra(
                                            ket=self.ions[model.target].levels.index(
                                                transition.level1
                                            ),
                                            bra=self.ions[model.target].levels.index(
                                                transition.level2
                                            ),
                                            subsystem=f"E{model.target}",
                                        )
                                        + KetBra(
                                            ket=self.ions[model.target].levels.index(
                                                transition.level2
                                            ),
                                            bra=self.ions[model.target].levels.index(
                                                transition.level1
                                            ),
                                            subsystem=f"E{model.target}",
                                        )
                                    )
                                    if i == model.target
                                    else Identity(subsystem=f"E{i}")
                                )
                                for i in range(self.N)
                            ],
                        )
                        @ displacement_plus
                    )
                )

                ops.append(
                    (
                        reduce(
                            lambda x, y: x @ y,
                            [
                                (
                                    WaveCoefficient(
                                        amplitude=rabi / 2,
                                        frequency=angular_frequency,
                                        phase=-model.phase,
                                    )
                                    * (
                                        KetBra(
                                            ket=self.ions[model.target].levels.index(
                                                transition.level1
                                            ),
                                            bra=self.ions[model.target].levels.index(
                                                transition.level2
                                            ),
                                            subsystem=f"E{model.target}",
                                        )
                                        + KetBra(
                                            ket=self.ions[model.target].levels.index(
                                                transition.level2
                                            ),
                                            bra=self.ions[model.target].levels.index(
                                                transition.level1
                                            ),
                                            subsystem=f"E{model.target}",
                                        )
                                    )
                                    if i == model.target
                                    else Identity(subsystem=f"E{i}")
                                )
                                for i in range(self.N)
                            ],
                        )
                        @ displacement_minus
                    )
                )

        else:
            for transition in self.ions[model.target].transitions:
                rabi = rabi_from_intensity(model, transition, I)

                ops.append(
                    reduce(
                        lambda x, y: x @ y,
                        [
                            (
                                WaveCoefficient(
                                    amplitude=rabi / 2,
                                    frequency=angular_frequency,
                                    phase=model.phase,
                                )
                                * (
                                    KetBra(
                                        ket=self.ions[model.target].levels.index(
                                            transition.level1
                                        ),
                                        bra=self.ions[model.target].levels.index(
                                            transition.level2
                                        ),
                                        subsystem=f"E{model.target}",
                                    )
                                    + KetBra(
                                        ket=self.ions[model.target].levels.index(
                                            transition.level2
                                        ),
                                        bra=self.ions[model.target].levels.index(
                                            transition.level1
                                        ),
                                        subsystem=f"E{model.target}",
                                    )
                                )
                                if i == model.target
                                else Identity(subsystem=f"E{i}")
                            )
                            for i in range(self.N)
                        ],
                    )
                )

        op = reduce(lambda x, y: x + y, ops)
        return op

    def map_Pulse(self, model, operands):
        return AtomicEmulatorGate(
            hamiltonian=operands["beam"],
            duration=model.duration,
        )

    def map_ParallelProtocol(self, model, operands):
        duration = operands["sequence"][0].duration

        for op in operands["sequence"]:
            assert op.duration == duration

        op = reduce(
            lambda x, y: x + y, map(lambda x: x.hamiltonian, operands["sequence"])
        )
        return AtomicEmulatorGate(hamiltonian=op, duration=duration)

    def map_SequentialProtocol(self, model, operands):
        return operands["sequence"]
