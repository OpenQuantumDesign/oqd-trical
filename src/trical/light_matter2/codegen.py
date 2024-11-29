from functools import reduce

import numpy as np

from oqd_core.interface.math import MathFunc, MathVar
from oqd_compiler_infrastructure import ConversionRule

########################################################################################

from .interface.atomic import SequentialProtocol

from .interface.operator import (
    KetBra,
    Annihilation,
    Creation,
    WaveCoefficient,
    Zero,
    Identity,
    Wave,
)
from .interface.emulator import AtomicEmulatorCircuit, AtomicEmulatorGate

from .utils import intensity_from_laser, rabi_from_intensity

########################################################################################


class ConstructHamiltonian(ConversionRule):
    def map_AtomicCircuit(self, model, operands):
        return AtomicEmulatorCircuit(
            base=operands["system"],
            sequence=(
                [operands["protocol"]]
                if isinstance(operands["protocol"], AtomicEmulatorGate)
                else operands["protocol"]
            ),
        )

    def map_System(self, model, operands):
        self.N = len(model.ions)
        self.M = len(model.modes)

        self.ions = model.ions
        self.modes = model.modes

        op = Zero()
        for n, ion in enumerate(operands["ions"]):
            op += reduce(
                lambda x, y: x @ y,
                [ion if i == n else Identity() for i in range(self.N + self.M)],
            )
        for m, mode in enumerate(operands["modes"]):
            op += reduce(
                lambda x, y: x @ y,
                [
                    mode if i == self.N + m else Identity()
                    for i in range(self.N + self.M)
                ],
            )

        return op

    def map_Ion(self, model, operands):
        op = Zero()
        for n, level in enumerate(model.levels):
            op += WaveCoefficient(
                amplitude=level.energy, frequency=0, phase=0
            ) * KetBra(ket=n, bra=n)

        return op

    def map_Phonon(self, model, operands):
        return WaveCoefficient(amplitude=model.energy, frequency=0, phase=0) * (
            Creation() * Annihilation()
        )

    def map_Beam(self, model, operands):
        op = Zero()

        I = intensity_from_laser(model)
        for transition in self.ions[model.target].transitions:
            rabi = rabi_from_intensity(model, transition, I)

            op += reduce(
                lambda x, y: x @ y,
                [
                    (
                        (
                            WaveCoefficient(
                                amplitude=rabi / 2,
                                frequency=abs(
                                    model.transition.level2.energy
                                    - model.transition.level1.energy
                                )
                                + model.detuning,
                                phase=model.phase,
                            )
                            + WaveCoefficient(
                                amplitude=rabi / 2,
                                frequency=-(
                                    abs(
                                        model.transition.level2.energy
                                        - model.transition.level1.energy
                                    )
                                    + model.detuning
                                ),
                                phase=-model.phase,
                            )
                        )
                        * (
                            KetBra(
                                ket=self.ions[model.target].levels.index(
                                    transition.level1
                                ),
                                bra=self.ions[model.target].levels.index(
                                    transition.level2
                                ),
                            )
                            + KetBra(
                                ket=self.ions[model.target].levels.index(
                                    transition.level2
                                ),
                                bra=self.ions[model.target].levels.index(
                                    transition.level1
                                ),
                            )
                        )
                        if i == model.target
                        else Identity()
                    )
                    for i in range(self.N)
                ],
            )

        op @= reduce(
            lambda x, y: x @ y,
            [
                Wave(
                    lamb_dicke=WaveCoefficient(
                        amplitude=np.dot(model.wavevector, mode.eigenvector)
                        * np.sqrt(1 / (2 * self.ions[model.target].mass * mode.energy)),
                        frequency=0,
                        phase=0,
                    )
                )
                for mode in self.modes
            ],
        )

        return op

    def map_Pulse(self, model, operands):
        return AtomicEmulatorGate(hamiltonian=operands["beam"], duration=model.duration)

    def map_ParallelProtocol(self, model, operands):
        # TODO: Implement correct procedure for SequentialProtocol
        # within ParallelProtocol
        for p in model.sequence:
            if isinstance(p, SequentialProtocol):
                raise NotImplementedError(
                    "SequentialProtocol within ParallelProtocol currently unsupported"
                )

        op = Zero()

        duration_max = np.max([_op.duration for _op in operands["sequence"]])

        for _op in operands["sequence"]:
            if _op.duration != duration_max:
                op += _op.hamiltonian * WaveCoefficient(
                    amplitude=MathFunc(
                        func="heaviside", expr=_op.duration - MathVar(name="t")
                    ),
                    frequency=0,
                    phase=0,
                )
            else:
                op += _op.hamiltonian

        return AtomicEmulatorGate(hamiltonian=op, duration=duration_max)

    def map_SequentialProtocol(self, model, operands):
        return operands["sequence"]
