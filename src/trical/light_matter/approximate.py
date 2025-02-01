from oqd_compiler_infrastructure import RewriteRule

from oqd_core.interface.math import MathNum

import numpy as np

########################################################################################

from .interface.operator import (
    Identity,
    ConstantCoefficient,
    Creation,
    Annihilation,
    WaveCoefficient,
)

########################################################################################


class FirstOrderLambDickeApprox(RewriteRule):
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
    def __init__(self):
        super().__init__()

        raise NotImplementedError

    def map_WaveCoefficient(self, model):
        pass


class RotatingReferenceFrame(RewriteRule):
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
