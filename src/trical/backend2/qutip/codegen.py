import qutip as qt

from oqd_compiler_infrastructure import ConversionRule

########################################################################################

from .interface import QutipExperiment

########################################################################################


class QutipCodeGeneration(ConversionRule):
    def __init__(self, fock_cutoff: int):
        super().__init__()

        self.fock_cutoff = fock_cutoff

    def map_Identity(self, model, operands):
        op = qt.identity(10)
        return QutipExperiment(hamiltonian=op, duration=1)

    def map_KetBra(self, model, operands):
        ket = qt.basis(10, model.ket)
        bra = qt.basis(10, model.bra).dag()
        op = ket * bra
        return QutipExperiment(hamiltonian=op, duration=1)

    def map_Annihilation(self, model, operands):
        op = qt.destroy(self.fock_cutoff)
        return QutipExperiment(hamiltonian=op, duration=1)

    def map_Creation(self, model, operands):
        op = qt.create(self.fock_cutoff)
        return QutipExperiment(hamiltonian=op, duration=1)

    def map_Wave(self, model, operands):
        op = (
            1j * 0.001 * (qt.create(self.fock_cutoff) + qt.destroy(self.fock_cutoff))
        ).expm()
        return QutipExperiment(hamiltonian=op, duration=1)

    def map_OperatorAdd(self, model, operands):
        op = operands["op1"].hamiltonian + operands["op2"].hamiltonian
        return QutipExperiment(hamiltonian=op, duration=1)

    def map_OperatorMul(self, model, operands):
        op = operands["op1"].hamiltonian * operands["op2"].hamiltonian
        return QutipExperiment(hamiltonian=op, duration=1)

    def map_OperatorScalarMul(self, model, operands):
        return operands["op"]

    def map_OperatorKron(self, model, operands):
        op = qt.tensor(operands["op1"].hamiltonian, operands["op2"].hamiltonian)
        return QutipExperiment(hamiltonian=op, duration=1)
