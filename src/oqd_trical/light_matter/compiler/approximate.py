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

from functools import cached_property, reduce

from oqd_compiler_infrastructure import RewriteRule
from oqd_core.interface.math import MathNum

########################################################################################
from oqd_trical.light_matter.interface.operator import (
    Annihilation,
    Creation,
    Identity,
    ConstantCoefficient,
    WaveCoefficient,
    KetBra,
)

########################################################################################


class FirstOrderLambDickeApprox(RewriteRule):
    """
    Applies the Lamb-Dicke approximation to first order.

    Attributes:
        cutoff (float): Lamb-Dicke parameter cutoff below which approximation is applied.
    """

    def __init__(self, cutoff=1):
        super().__init__()
        self.cutoff = cutoff

        self.approximated_operators = []

    def map_Displacement(self, model):
        if isinstance(model.alpha.amplitude, MathNum):
            if np.abs(model.alpha.amplitude.value) < self.cutoff:
                self.approximated_operators.append(model)

                alpha_conj = WaveCoefficient(
                    amplitude=model.alpha.amplitude,
                    frequency=-model.alpha.frequency,
                    phase=-model.alpha.phase,
                )
                return Identity(subsystem=model.subsystem) + (
                    model.alpha * Creation(subsystem=model.subsystem)
                    - alpha_conj * Annihilation(subsystem=model.subsystem)
                )


class SecondOrderLambDickeApprox(RewriteRule):
    """
    Applies the Lamb-Dicke approximation to second order.

    Attributes:
        cutoff (float): Lamb-Dicke parameter cutoff below which approximation is applied.
    """

    def __init__(self, cutoff=1):
        super().__init__()

        self.cutoff = cutoff

        self.approximated_operators = []

    def map_Displacement(self, model):
        if isinstance(model.alpha.amplitude, MathNum):
            if np.abs(model.alpha.amplitude.value) < self.cutoff:
                self.approximated_operators.append(model)

                alpha_conj = WaveCoefficient(
                    amplitude=model.alpha.amplitude,
                    frequency=-model.alpha.frequency,
                    phase=-model.alpha.phase,
                )
                return (
                    Identity(subsystem=model.subsystem)
                    + (
                        model.alpha * Creation(subsystem=model.subsystem)
                        - alpha_conj * Annihilation(subsystem=model.subsystem)
                    )
                    + (
                        model.alpha * Creation(subsystem=model.subsystem)
                        - alpha_conj * Annihilation(subsystem=model.subsystem)
                    )
                    * (
                        model.alpha * Creation(subsystem=model.subsystem)
                        - alpha_conj * Annihilation(subsystem=model.subsystem)
                    )
                )


class RotatingWaveApprox(RewriteRule):
    """
    Applies the rotating wave approximation.

    Attributes:
        cutoff (float): Frequency cutoff above which approximation is applied.

    Warning:
        Currently not implmented!
    """

    def __init__(self):
        super().__init__()

        raise NotImplementedError

    def map_WaveCoefficient(self, model):
        return (
            WaveCoefficient(
                amplitude=1, frequency=self.frame_specs[model.subsystem], phase=0
            )
            * model
        )


########################################################################################


class RotatingReferenceFrame(RewriteRule):
    """
    Moves to an interaction picture with a rotating frame of reference.

    Attributes:
        frame (Operator): [`Operator`][oqd_trical.light_matter.interface.operator.Operator] that defines the rotating frame of reference.

    Warning:
        Currently not implmented!
    """

    def __init__(self, frame_specs):
        super().__init__()

        self.frame_specs = frame_specs

    @cached_property
    def system(self):
        return list(self.frame_specs.keys())

    def _complete_operator(self, op, subsystem):
        return reduce(
            lambda x, y: x @ y,
            [op if subsystem == s else Identity(subsystem=s) for s in self.system],
        )

    @cached_property
    def frame(self):
        ops = []
        for subsystem, energy in self.frame_specs.items():
            if subsystem[0] == "E":
                ops.extend(
                    [
                        ConstantCoefficient(value=e)
                        * self._complete_operator(
                            KetBra(ket=n, bra=n, subsystem=subsystem), subsystem
                        )
                        for n, e in enumerate(energy)
                    ]
                )
            else:
                ops.append(
                    ConstantCoefficient(value=energy)
                    * self._complete_operator(
                        Creation(subsystem=subsystem)
                        * Annihilation(subsystem=subsystem),
                        subsystem,
                    )
                )

        return reduce(lambda x, y: x + y, ops)

    def map_AtomicEmulatorCircuit(self, model):
        return model.__class__(
            frame=self.frame, base=model.base - self.frame, sequence=model.sequence
        )

    def map_KetBra(self, model):
        return (
            WaveCoefficient(
                amplitude=1,
                frequency=self.frame_specs[model.subsystem][model.ket]
                - self.frame_specs[model.subsystem][model.bra],
                phase=0,
            )
            * model
        )

    def map_Displacement(self, model):
        alpha = model.alpha
        return model.__class__(
            alpha=alpha.__class__(
                amplitude=alpha.amplitude,
                frequency=alpha.frequency + self.frame_specs[model.subsystem],
                phase=alpha.phase,
            ),
            subsystem=model.subsystem,
        )

    def map_Annihilation(self, model):
        return (
            WaveCoefficient(
                amplitude=1, frequency=self.frame_specs[model.subsystem], phase=0
            )
            * model
        )

    def map_Creation(self, model):
        return (
            WaveCoefficient(
                amplitude=1, frequency=-self.frame_specs[model.subsystem], phase=0
            )
            * model
        )
