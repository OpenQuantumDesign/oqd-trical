
import qutip as qt
import matplotlib.pyplot as plt

########################################################################################

from typing import List, Dict, Literal, Tuple, Union
from pydantic import ConfigDict
from pydantic.types import NonNegativeInt
from oqd_compiler_infrastructure import VisitableBaseModel

########################################################################################

from oqd_core.interface.math import MathExpr
from oqd_core.backend.metric import *

########################################################################################

__all__ = ["QutipOperation", "QutipExperiment", "TaskArgsQutip", "QutipExpectation"]


class QutipExpectation(VisitableBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    operator: List[Tuple[qt.Qobj, MathExpr]]


class TaskArgsQutip(VisitableBaseModel):
    """
    Class representing args for QuTip

    Attributes:
        layer (str): the layer of the experiment (analog, atomic)
        n_shots (Union[int, None]): number of shots requested
        fock_cutof (int): fock_cutoff for QuTip simulation
        dt (float): timesteps for discrete time
        metrics (dict): metrics which should be computed for the experiment. This does not require any Measure instruction in the analog layer.
    """

    layer: Literal["analog"] = "analog"
    n_shots: Union[int, None] = 10
    fock_cutoff: int = 4
    dt: float = 0.1
    metrics: Dict[
        str, Union[EntanglementEntropyReyni, EntanglementEntropyVN, QutipExpectation]
    ] = {}


class QutipOperation(VisitableBaseModel):
    """
    Class representing a quantum operation in QuTip

    Attributes:
        hamiltonian (List[qt.Qobj, str]): Hamiltonian to evolve by
        duration (float): Duration of the evolution
        
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    hamiltonian: List[Tuple[qt.Qobj, MathExpr]]
    duration: float


class QutipMeasurement(VisitableBaseModel):
    pass


class QutipExperiment(VisitableBaseModel):
    """
    Class representing a quantum experiment in qutip

    Attributes:
        instructions (List[QutipOperation]): List of quantum operations to apply
        n_qreg (NonNegativeInt): Number of qubit quantum registers
        n_qmode (NonNegativeInt): Number of modal quantum registers
        args (TaskArgsQutip): Arguments for the experiment
    """

    instructions: list[Union[QutipOperation, QutipMeasurement]]
    n_qreg: NonNegativeInt
    n_qmode: NonNegativeInt

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

    def quick_plot(self):
        """Convenience method for plotting expectation values for a Results object"""

        for name in self.expectation_values:
            plt.plot(self.times, self.expectation_values[name], label=name)

        plt.xlim(self.times[0], self.times[-1])
        y_min = min(
            [min(self.expectation_values[name]) for name in self.expectation_values]
        )
        y_max = max(
            [max(self.expectation_values[name]) for name in self.expectation_values]
        )
        plt.ylim(y_min, y_max)
        prefixes = {
            1: r"s",
            1e-3: r"ms",
            1e-6: r"$\mu s$",
            1e-9: r"ns",
        }

        if self.timescale in prefixes:
            prefix = prefixes[self.timescale]
        else:
            prefix = f"{self.timescale} s"

        plt.xlabel(f"t ({prefix})")
        plt.grid()
        plt.legend()
        plt.show()
