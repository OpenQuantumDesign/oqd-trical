import numpy as np
import itertools
import time
import qutip as qt

from cmath import exp
from trical.light_matter.utilities import displace
from qutip import qeye, basis, tensor, destroy, QobjEvo, Qobj, mesolve
from oqd_compiler_infrastructure import ConversionRule, RewriteRule
from oqd_core.compiler.math.passes import evaluate_math_expr, simplify_math_expr

########################################################################################

__all__ = [
    "entanglement_entropy_vn",
    "QutipMetricConversion",
    "QutipBackendCompiler",
    "QutipExperimentVM",
]

########################################################################################

class Results:
    """Class for packaging results obtained from Qutip's solver

    Args:
        qutip_results (qutip.results): result object from qutip
        ops (list): list of operators for which expectation values were taken
        times (iterable): list of times to evaluate these expectation values
        timescale (float): time unit (e.g. 1e-6 for microseconds)

    Attributes:
        expectation_values(dict): Dictionary mapping name of QuantumOperator object to expectation values from Qutip
    """

    def __init__(self, qutip_results, ops, times, timescale):
        self.qutip_results = qutip_results
        self.ops = ops
        self.times = times
        self.timescale = timescale

        self.expectation_values = {}
        for n, ion_proj in enumerate(self.ops):
            self.expectation_values[ion_proj.name] = self.qutip_results.expect[n]

def entanglement_entropy_vn(t, psi, qreg, qmode, n_qreg, n_qmode):
    rho = qt.ptrace(
        psi,
        qreg + [n_qreg + m for m in qmode],
    )
    return qt.entropy_vn(rho)


class QutipMetricConversion(ConversionRule):
    """
    This takes in a a dictionary containing Metrics, which get converted to lambda functions for QuTip

    Args:
        model (dict): The values are Analog layer Operators

    Returns:
        model (dict): The values are lambda functions

    Note:
        n_qreg and n_qmode are given as compiler parameters
    """

    def __init__(self, n_qreg, n_qmode):
        super().__init__()
        self._n_qreg = n_qreg
        self._n_qmode = n_qmode

    def map_QutipExpectation(self, model, operands):
        for idx, operator in enumerate(model.operator):
            coefficient = evaluate_math_expr(operator[1])
            op_exp = (
                coefficient * operator[0]
                if idx == 0
                else op_exp + coefficient * operator[0]
            )
        return lambda t, psi: qt.expect(op_exp, psi)

    def map_EntanglementEntropyVN(self, model, operands):
        return lambda t, psi: entanglement_entropy_vn(
            t, psi, model.qreg, model.qmode, self._n_qreg, self._n_qmode
        )


class QutipExperimentVM(RewriteRule):
    """
    This is a Virtual Machine which takes in a QutipExperiment object, simulates the experiment and then produces the results

    Args:
        model (QutipExperiment): This is the compiled  [`QutipExperiment`][oqd_analog_emulator.qutip_backend.QutipExperiment] object

    Returns:
        task (TaskResultAnalog):

    Note:
        n_qreg and n_qmode are given as compiler parameters
    """

    def __init__(self, qt_metrics, n_shots, fock_cutoff, dt):
        super().__init__()
        self.results = Results(runtime=0)
        self._qt_metrics = qt_metrics
        self._n_shots = n_shots
        self._fock_cutoff = fock_cutoff
        self._dt = dt

    def map_QutipExperiment(self, model):

        dims = model.n_qreg * [2] + model.n_qmode * [self._fock_cutoff]
        self.n_qreg = model.n_qreg
        self.n_qmode = model.n_qmode
        self.current_state = qt.tensor([qt.basis(d, 0) for d in dims])

        self.results.times.append(0.0)
        self.results.state = list(
            self.current_state.full().squeeze(),
        )
        self.results.metrics.update(
            {
                key: [self._qt_metrics[key](0.0, self.current_state)]
                for key in self._qt_metrics.keys()
            }
        )

    def map_QutipMeasurement(self, model):
        if self._n_shots is None:
            self.results.counts = {}
        else:
            probs = np.power(np.abs(self.current_state.full()), 2).squeeze()
            n_shots = self._n_shots
            inds = np.random.choice(len(probs), size=n_shots, p=probs)
            opts = self.n_qreg * [[0, 1]] + self.n_qmode * [
                list(range(self._fock_cutoff))
            ]
            bases = list(itertools.product(*opts))
            shots = np.array([bases[ind] for ind in inds])
            bitstrings = ["".join(map(str, shot)) for shot in shots]
            self.results.counts = {
                bitstring: bitstrings.count(bitstring) for bitstring in bitstrings
            }

        self.results.state = list(
            self.current_state.full().squeeze(),
        )

    def map_QutipOperation(self, model):

        duration = model.duration
        tspan = np.linspace(0, duration, round(duration / self._dt)).tolist()

        qutip_hamiltonian = []
        for op, coeff in model.hamiltonian:
            qutip_hamiltonian.append([op, simplify_math_expr(coeff)])

        start_runtime = time.time()
        result_qobj = qt.sesolve(
            qutip_hamiltonian,
            self.current_state,
            tspan,
            e_ops=self._qt_metrics,
            options={"store_states": True},
        )
        self.results.runtime = time.time() - start_runtime + self.results.runtime

        self.results.times.extend([t + self.results.times[-1] for t in tspan][1:])

        for idx, key in enumerate(self.results.metrics.keys()):
            self.results.metrics[key].extend(result_qobj.expect[idx].tolist()[1:])

        self.current_state = result_qobj.final_state

        self.results.state = list(
            result_qobj.final_state.full().squeeze(),
        )


class QutipBackendCompiler(ConversionRule):
    """
    This is a ConversionRule which compiles analog layer objects to QutipExperiment objects

    Args:
        model (VisitableBaseModel): This takes in objects in Analog level and converts them to representations which can be used to run QuTip simulations.

    Returns:
        model (Union[VisitableBaseModel, Any]): QuTip objects and representations which can be used to run QuTip simulations

    """

    def __init__(self, fock_cutoff=None):
        super().__init__()
        self._fock_cutoff = fock_cutoff

    """ReWrite rule for converting a Hamiltonian tree object to a time-dependent QuTiP object (QobjEvo)

    Attributes:
        operands (list): operand stack used for converting between representations

    """

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
