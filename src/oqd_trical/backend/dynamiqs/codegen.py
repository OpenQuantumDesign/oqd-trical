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

import dynamiqs as dq
from jax import numpy as jnp
from oqd_compiler_infrastructure import ConversionRule

########################################################################################
from oqd_trical.backend.dynamiqs.interface import DynamiqsExperiment, DynamiqsGate
from oqd_trical.light_matter.compiler.analysis import HilbertSpace
from oqd_trical.light_matter.interface.operator import PrunedOperator

########################################################################################


class DynamiqsCodeGeneration(ConversionRule):
    """
    Rule that converts an [`AtomicEmulatorCircuit`][oqd_trical.light_matter.interface.emulator.AtomicEmulatorCircuit]
    to a [`DynamiqsExperiment`][oqd_trical.backend.dynamiqs.interface.DynamiqsExperiment]

    Attributes:
        hilbert_space (Dict[str, int]): Hilbert space of the system.
    """

    def __init__(self, hilbert_space: HilbertSpace):
        super().__init__()

        self.hilbert_space = hilbert_space

    def map_AtomicEmulatorCircuit(self, model, operands):
        return DynamiqsExperiment(
            frame=None
            if (
                isinstance(operands["frame"], PrunedOperator)
                or operands["frame"] is None
            )
            else dq.timecallable(operands["frame"]),
            sequence=operands["sequence"],
        )

    def map_AtomicEmulatorGate(self, model, operands):
        if isinstance(operands["hamiltonian"], PrunedOperator):
            return DynamiqsGate(hamiltonian=None, duration=operands["duration"])

        return DynamiqsGate(
            hamiltonian=dq.timecallable(operands["hamiltonian"]),
            duration=operands["duration"],
        )

    def map_Identity(self, model, operands):
        op = dq.eye(self.hilbert_space.size[model.subsystem])
        return lambda t: op

    def map_KetBra(self, model, operands):
        ket = dq.basis(self.hilbert_space.size[model.subsystem], model.ket)
        bra = dq.basis(self.hilbert_space.size[model.subsystem], model.bra).dag()
        op = ket @ bra

        if not isinstance(op, dq.QArray):
            op = dq.asqarray(op)
        return lambda t: op

    def map_Annihilation(self, model, operands):
        op = dq.destroy(self.hilbert_space.size[model.subsystem])
        return lambda t: op

    def map_Creation(self, model, operands):
        op = dq.create(self.hilbert_space.size[model.subsystem])
        return lambda t: op

    def map_Displacement(self, model, operands):
        return lambda t: dq.displace(
            self.hilbert_space.size[model.subsystem], operands["alpha"](t)
        )

    def map_OperatorMul(self, model, operands):
        return lambda t: operands["op1"](t) @ operands["op2"](t)

    def map_OperatorKron(self, model, operands):
        return lambda t: dq.tensor(operands["op1"](t), operands["op2"](t))

    def map_OperatorAdd(self, model, operands):
        return lambda t: operands["op1"](t) + operands["op2"](t)

    def map_OperatorScalarMul(self, model, operands):
        return lambda t: operands["coeff"](t) * operands["op"](t)

    def map_WaveCoefficient(self, model, operands):
        return lambda t: operands["amplitude"](t) * jnp.exp(
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
            return lambda t: getattr(jnp, model.func)(operands["expr"](t))

        if model.func == "heaviside":
            return lambda t: jnp.heaviside(operands["expr"](t), 1)

        if model.func == "conj":
            return lambda t: jnp.conj(operands["expr"](t))

        if model.func == "abs":
            return lambda t: jnp.abs(operands["expr"](t))

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
