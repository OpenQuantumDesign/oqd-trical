import qutip as qt
import numpy as np

from oqd_compiler_infrastructure import ConversionRule

########################################################################################

from .interface import QutipExperiment, QutipGate

########################################################################################


class QutipCodeGeneration(ConversionRule):
    def __init__(self, fock_cutoff: int):
        super().__init__()

        self.fock_cutoff = fock_cutoff

    def map_AtomicEmulatorCircuit(self, model, operands):
        return QutipExperiment(base=operands["base"], sequence=operands["sequence"])

    def map_AtomicEmulatorGate(self, model, operands):
        return QutipGate(
            hamiltonian=operands["hamiltonian"], duration=operands["duration"]
        )

    def map_Identity(self, model, operands):
        op = qt.identity(10)
        return lambda t: op

    def map_KetBra(self, model, operands):
        ket = qt.basis(10, model.ket)
        bra = qt.basis(10, model.bra).dag()
        op = ket * bra
        return lambda t: op

    def map_Annihilation(self, model, operands):
        op = qt.destroy(self.fock_cutoff)
        return lambda t: op

    def map_Creation(self, model, operands):
        op = qt.create(self.fock_cutoff)
        return lambda t: op

    def map_Wave(self, model, operands):
        op = (
            1j * 0.001 * (qt.create(self.fock_cutoff) + qt.destroy(self.fock_cutoff))
        ).expm()
        return lambda t: op

    def map_OperatorMul(self, model, operands):
        return lambda t: operands["op1"](t) * operands["op2"](t)

    def map_OperatorKron(self, model, operands):
        return lambda t: qt.tensor(operands["op1"](t), operands["op2"](t))

    def map_OperatorScalarMul(self, model, operands):
        return lambda t: operands["coeff"](t) * operands["op"](t)

    def map_OperatorAdd(self, model, operands):
        return lambda t: operands["op1"](t) + operands["op2"](t)

    def map_WaveCoefficient(self, model, operands):
        return lambda t: operands["amplitude"] * np.exp(
            1j * (operands["frequency"] * t + operands["phase"])
        )

    def map_MathNum(self, model, operands):
        return model.value

    def map_CoefficientAdd(self, model, operands):
        return lambda t: operands["coeff1"](t) + operands["coeff2"](t)

    def map_CoefficientMul(self, model, operands):
        return lambda t: operands["coeff1"](t) * operands["coeff2"](t)
