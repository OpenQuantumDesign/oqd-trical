import qutip as qt

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
        return op

    def map_KetBra(self, model, operands):
        ket = qt.basis(10, model.ket)
        bra = qt.basis(10, model.bra).dag()
        op = ket * bra
        return op

    def map_Annihilation(self, model, operands):
        op = qt.destroy(self.fock_cutoff)
        return op

    def map_Creation(self, model, operands):
        op = qt.create(self.fock_cutoff)
        return op

    def map_Wave(self, model, operands):
        op = (
            1j * 0.001 * (qt.create(self.fock_cutoff) + qt.destroy(self.fock_cutoff))
        ).expm()
        return op

    def map_OperatorAdd(self, model, operands):
        op = operands["op1"] + operands["op2"]
        return op

    def map_OperatorMul(self, model, operands):
        op = operands["op1"] * operands["op2"]
        return op

    def map_OperatorScalarMul(self, model, operands):
        return operands["op"]

    def map_OperatorKron(self, model, operands):
        op = qt.tensor(operands["op1"], operands["op2"])
        return op
