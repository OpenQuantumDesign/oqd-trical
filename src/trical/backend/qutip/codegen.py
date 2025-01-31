import qutip as qt
import numpy as np
import math

from typing import Dict

from oqd_compiler_infrastructure import ConversionRule

########################################################################################

from .interface import QutipExperiment, QutipGate

########################################################################################


class QutipCodeGeneration(ConversionRule):
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

    def map_Wave(self, model, operands):
        def f_op(t):
            lamb_dicke = operands["lamb_dicke"](t)
            return (
                1j
                * (
                    lamb_dicke * qt.create(self.hilbert_space[model.subsystem])
                    + np.conj(lamb_dicke)
                    * qt.destroy(self.hilbert_space[model.subsystem])
                )
            ).expm()

        return f_op

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
