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

import math
from typing import Dict

import numpy as np
import qutip as qt
from oqd_compiler_infrastructure import ConversionRule

########################################################################################
from .interface import QutipExperiment, QutipGate

########################################################################################


class QutipCodeGeneration(ConversionRule):
    """
    Rule that converts an [`AtomicEmulatorCircuit`][oqd_trical.light_matter.interface.emulator.AtomicEmulatorCircuit]
    to a [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment]

    Attributes:
        hilbert_space (Dict[str, int]): Hilbert space of the system.
    """

    def __init__(self, hilbert_space: Dict[str, int]):
        super().__init__()

        self.hilbert_space = hilbert_space

    def map_AtomicEmulatorCircuit(self, model, operands):
        return QutipExperiment(base=operands["base"], sequence=operands["sequence"])

    def map_AtomicEmulatorGate(self, model, operands):
        return QutipGate(
            hamiltonian=operands["hamiltonian"], duration=operands["duration"]
        )

    def map_Identity(self, model, operands):
        op = qt.identity(self.hilbert_space[model.subsystem])
        return lambda t: op

    def map_KetBra(self, model, operands):
        ket = qt.basis(self.hilbert_space[model.subsystem], model.ket)
        bra = qt.basis(self.hilbert_space[model.subsystem], model.bra).dag()
        op = ket * bra
        return lambda t: op

    def map_Annihilation(self, model, operands):
        op = qt.destroy(self.hilbert_space[model.subsystem])
        return lambda t: op

    def map_Creation(self, model, operands):
        op = qt.create(self.hilbert_space[model.subsystem])
        return lambda t: op

    def map_Displacement(self, model, operands):
        return lambda t: qt.displace(
            self.hilbert_space[model.subsystem], operands["alpha"](t)
        )

    def map_OperatorMul(self, model, operands):
        return lambda t: operands["op1"](t) * operands["op2"](t)

    def map_OperatorKron(self, model, operands):
        return lambda t: qt.tensor(operands["op1"](t), operands["op2"](t))

    def map_OperatorAdd(self, model, operands):
        return lambda t: operands["op1"](t) + operands["op2"](t)

    def map_OperatorScalarMul(self, model, operands):
        return lambda t: operands["coeff"](t) * operands["op"](t)

    def map_WaveCoefficient(self, model, operands):
        return lambda t: operands["amplitude"](t) * np.exp(
            1j * (operands["frequency"](t) * t + operands["phase"](t))
        )

    def map_CoefficientAdd(self, model, operands):
        return lambda t: operands["coeff1"](t) + operands["coeff2"](t)

    def map_CoefficientMul(self, model, operands):
        return lambda t: operands["coeff1"](t) * operands["coeff2"](t)

    def map_MathNum(self, model, operands):
        return lambda t: model.value

    def map_MathImag(self, model, operands):
        return lambda t: 1j

    def map_MathVar(self, model, operands):
        if model.name == "t":
            return lambda t: t

        raise ValueError(
            f"Unsupported variable {model.name}, only variable t is supported"
        )

    def map_MathFunc(self, model, operands):
        if getattr(math, model.func, None):
            return lambda t: getattr(math, model.func)(operands["expr"](t))

        if model.func == "heaviside":
            return lambda t: np.heaviside(operands["expr"](t), 1)

        if model.func == "conj":
            return lambda t: np.conj(operands["expr"](t))

        raise ValueError(f"Unsupported function {model.func}")

    def map_MathAdd(self, model, operands):
        return lambda t: operands["expr1"](t) + operands["expr2"](t)

    def map_MathSub(self, model, operands):
        return lambda t: operands["expr1"](t) - operands["expr2"](t)

    def map_MathMul(self, model, operands):
        return lambda t: operands["expr1"](t) * operands["expr2"](t)

    def map_MathDiv(self, model, operands):
        return lambda t: operands["expr1"](t) / operands["expr2"](t)

    def map_MathPow(self, model, operands):
        return lambda t: operands["expr1"](t) ** operands["expr2"](t)
