from functools import reduce

import numpy as np

from oqd_compiler_infrastructure import ConversionRule

########################################################################################

from .interface.operator import (
    KetBra,
    Annihilation,
    Creation,
    WaveCoefficient,
    Zero,
    Identity,
    Wave,
)

from .utils import intensity, rabi_from_intensity

########################################################################################


class ConstructHamiltonian(ConversionRule):
    def map_AtomicCircuit(self, model, operands):
        return operands["system"] + operands["protocol"]

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

        I = intensity(model)
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
        return operands["beam"]

    def map_ParallelProtocol(self, model, operands):
        return reduce(lambda x, y: x + y, operands["sequence"])

    def map_SeqeuntialProtocol(self, model, operands):
        return reduce(lambda x, y: x + y, operands["sequence"])
