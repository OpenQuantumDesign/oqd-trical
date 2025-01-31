from oqd_compiler_infrastructure import RewriteRule

from oqd_core.interface.math import MathNum

import numpy as np

########################################################################################

from .interface.operator import (
    Identity,
    ConstantCoefficient,
    Creation,
    Annihilation,
)

########################################################################################


class FirstOrderLambDickeApprox(RewriteRule):
    def __init__(self, cutoff=1):
        super().__init__()
        self.cutoff = cutoff

        self.approximated_operators = []

    def map_Wave(self, model):
        if isinstance(model.lamb_dicke.amplitude, MathNum):
            if np.abs(model.lamb_dicke.amplitude.value) < self.cutoff:
                self.approximated_operators.append(model)
                return Identity(subsystem=model.subsystem) + ConstantCoefficient(
                    value=1j
                ) * (
                    model.lamb_dicke * Creation(subsystem=model.subsystem)
                    + model.lamb_dicke * Annihilation(subsystem=model.subsystem)
                )


class SecondOrderLambDickeApprox(RewriteRule):
    def __init__(self, cutoff=1):
        super().__init__()

        self.cutoff = cutoff

        self.approximated_operators = []

    def map_Wave(self, model):
        if isinstance(model.lamb_dicke.amplitude, MathNum):
            if np.abs(model.lamb_dicke.amplitude.value) < self.cutoff:
                self.approximated_operators.append(model)
                return (
                    Identity(subsystem=model.subsystem)
                    + ConstantCoefficient(value=1j)
                    * (
                        model.lamb_dicke * Creation(subsystem=model.subsystem)
                        + model.lamb_dicke * Annihilation(subsystem=model.subsystem)
                    )
                    + ConstantCoefficient(value=-1)
                    * (
                        model.lamb_dicke * Creation(subsystem=model.subsystem)
                        + model.lamb_dicke * Annihilation(subsystem=model.subsystem)
                    )
                    * (
                        model.lamb_dicke * Creation(subsystem=model.subsystem)
                        + model.lamb_dicke * Annihilation(subsystem=model.subsystem)
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
