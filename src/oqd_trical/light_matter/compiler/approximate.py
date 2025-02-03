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

from oqd_compiler_infrastructure import RewriteRule

from oqd_core.interface.math import MathNum

import numpy as np

########################################################################################

from ..interface.operator import (
    Identity,
    Creation,
    Annihilation,
    WaveCoefficient,
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
        pass


class RotatingReferenceFrame(RewriteRule):
    """
    Moves to an interaction picture with a rotating frame of reference.

    Attributes:
        frame (Operator): [`Operator`][oqd_trical.light_matter.interface.operator.Operator] that defines the rotating frame of reference.

    Warning:
        Currently not implmented!
    """

    def __init__(self, frame):
        super().__init__()

        self.frame = frame

        raise NotImplementedError

    def map_KetBra(self, model):
        pass

    def map_Wave(self, model):
        pass

    def map_Annihilation(self, model):
        pass

    def map_Creation(self, model):
        pass
