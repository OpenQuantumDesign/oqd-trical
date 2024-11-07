from cmath import exp
from qutip import qeye, basis, tensor, destroy, QobjEvo, Qobj, mesolve
from oqd_compiler_infrastructure import RewriteRule 
from trical.light_matter.utilities import displace
from oqd_compiler_infrastructure import Post, Pre 
from trical.backend.qutip.conversion import (
    QutipBackendCompiler as CompilerBackend,
    QutipExperimentVM as ExperimentViewModel,
    QutipMetricConversion as MetricConversion,
)

__all__ = [
    "entanglement_entropy_vn",
    "MetricConversion",
    "CompilerBackend",
    "ExperimentViewModel",
]

def time_evolve(H, psi_0, times, e_ops=None, progress_bar=True):
    """Function for time-evolving a quantum state using QuTiP's mesolve.

    Attributes:
        operands (list): Operand stack used for time evolution.
    """

    return mesolve(
        H,
        psi_0,
        times,
        e_ops=e_ops,
        options={
            "progress_bar": progress_bar,
            "store_final_state": True,
        },
    )

class QuantumOperator(RewriteRule):
    """
    Class representing quantum operators for expectation values.

    Attributes:
        qobj (Qobj): Qutip-compatible object.
        name (str): Alias/name for the operator.
    """

    qobj: Qobj
    name: str

    class Config:
        arbitrary_types_allowed = True

class QutipConversion(RewriteRule):
    """ReWrite rule for converting a Hamiltonian tree object to a time-dependent QuTiP object (QobjEvo)

    Attributes:
        operands (list): operand stack used for converting between representations

    """

    def __init__(self, ld_order=1):
        super().__init__()
        self.operands = []

    def map_ApproxDisplacementMatrix(self, model):

        alpha = self.operands.pop()

        op = displace(
            ld_order=model.ld_order,
            alpha=alpha,
            rwa_cutoff=model.rwa_cutoff,
            Delta=model.Delta,
            nu=model.nu,
            dims=model.dims,
        )

        self.operands.append(op)

    def map_KetBra(self, model):
        """Function mapping KetBra objects to a Qutip projector
        Attributes:
            model (KetBra): KetBra object |iXj|
        """

        op = (
            basis(model.dims, model.dims - model.i - 1)
            * basis(model.dims, model.dims - model.j - 1).dag()
        )
        self.operands.append(op)

        pass

    def map_Annihilation(self, model):
        """Function mapping Annhilation objects to a Qutip destroy object

        Attributes:
            model (Annhilation):
        """
        op = destroy(model.dims)
        self.operands.append(op)
        pass

    def map_Creation(self, model):
        """Function mapping Creation objects to a Qutip destroy.dag() object

        Attributes:
            model (Creation):
        """
        op = destroy(model.dims).dag()
        self.operands.append(op)
        pass

    def map_Identity(self, model):
        """Function mapping Identity objects to a Qutip qeye() object

        Attributes:
            model (Identity):
        """
        op = qeye(model.dims)
        self.operands.append(op)
        pass

    def map_OperatorAdd(self, model):

        if len(self.operands) == 1:
            # Case where we have an op and its hc

            op1 = self.operands.pop()

            def time_dep_fn(t, args={}):
                return op1(t, args) + op1(t, args).dag()

            op = QobjEvo(time_dep_fn)
        else:

            op2 = self.operands.pop()
            op1 = self.operands.pop()

            if isinstance(op1, QobjEvo) and isinstance(op2, QobjEvo):

                def time_dep_fn(t, args={}):
                    return op1(t, args) + op2(t, args)

                op = QobjEvo(time_dep_fn)
            else:
                op = op1 + op2

        self.operands.append(op)
        pass

    def map_OperatorMul(self, model):
        op2 = self.operands.pop()
        op1 = self.operands.pop()

        if isinstance(op1, QobjEvo) and isinstance(op2, QobjEvo):

            def time_dep_fn(t, args={}):
                return op1(t, args) * op2(t, args)

            op = QobjEvo(time_dep_fn)
        else:

            op = op1 @ op2
        self.operands.append(op)
        pass

    def map_OperatorKron(self, model):

        op2 = self.operands.pop()
        op1 = self.operands.pop()
        op = tensor(op1, op2)

        self.operands.append(op)

        pass

    def map_OperatorScalarMul(self, model):
        coeff = self.operands.pop()
        op = self.operands.pop()

        if callable(coeff):  # map returned some function of t and args

            time_dep_fn = lambda t, args={}: coeff(t, args) * qeye(op.dims[1])
            coeffp = QobjEvo(time_dep_fn)
        else:
            coeffp = coeff

        self.operands.append(coeffp * op)
        pass

    def map_CoefficientAdd(self, model):
        coeff2 = self.operands.pop()
        coeff1 = self.operands.pop()

        if callable(coeff1) and not callable(coeff2):

            def time_dep_fn(t, args={}):
                return coeff1(t, args) + coeff2

            op = time_dep_fn

        elif callable(coeff2) and not callable(coeff1):

            def time_dep_fn(t, args={}):
                return coeff1 + coeff2(t, args)

            op = time_dep_fn
        elif callable(coeff1) and callable(coeff2):

            def time_dep_fn(t, args={}):
                return coeff1(t, args) + coeff2(t, args)

            op = time_dep_fn
        else:
            op = coeff1 + coeff2

        self.operands.append(op)
        pass

    def map_CoefficientMul(self, model):
        coeff2 = self.operands.pop()
        coeff1 = self.operands.pop()

        if callable(coeff1) and not callable(coeff2):

            def time_dep_fn(t, args={}):
                return coeff1(t, args) * coeff2

            op = time_dep_fn

        elif callable(coeff2) and not callable(coeff1):

            def time_dep_fn(t, args={}):
                return coeff1 * coeff2(t, args)

            op = time_dep_fn
        elif callable(coeff1) and callable(coeff2):

            def time_dep_fn(t, args={}):
                return coeff1(t, args) * coeff2(t, args)

            op = time_dep_fn
        else:
            op = coeff1 * coeff2

        self.operands.append(op)
        pass

    def map_ConstantCoefficient(self, model):
        self.operands.append(model.value)
        pass

    def map_WaveCoefficient(self, model):
        A = model.amplitude
        omega = model.frequency
        phi = model.phase

        time_dep_fn = lambda t, args={}: A * exp(1j * (omega * t + phi))

        self.operands.append(time_dep_fn)

        pass

    def compiler_circuit_to_backendIR(model, fock_cutoff):
        """
        This compiles ([`Circuit`][your_project.interface.circuit.Circuit] to a list of [`BackendOperation`][your_project.interface.BackendOperation] objects

        Args:
            model (Circuit):
            fock_cutoff (int): fock_cutoff for Ladder Operators

        Returns:
            (list(BackendOperation)):
        """
        return Post(CompilerBackend(fock_cutoff=fock_cutoff))(model=model)


    def compiler_args_to_backendIR(model):
        """
        This compiles TaskArgs to a list of TaskArgsBackend

        Args:
            model (TaskArgs):

        Returns:
            (TaskArgsBackend):
        """
        return Post(CompilerBackend(fock_cutoff=model.fock_cutoff))(model=model)


    def run_backend_experiment(model: ExperimentViewModel, args):
        """
        This takes in a [`BackendExperiment`][your_project.interface.BackendExperiment] and produces a TaskResult object

        Args:
            model (BackendExperiment):
            args: (BackendArgs)

        Returns:
            (TaskResult): Contains results of the simulation
        """
        n_qubit = model.n_qubit
        n_mode = model.n_mode
        metrics = Post(MetricConversion(n_qubit=n_qubit, n_mode=n_mode))(args.metrics)
        interpreter = Pre(
            ExperimentViewModel(
                backend_metrics=metrics,
                n_shots=args.n_shots,
                fock_cutoff=args.fock_cutoff,
                dt=args.dt,
            )
        )
        interpreter(model=model)

        return interpreter.children[0].results 

