from __future__ import annotations

from typing import List, Union, Callable

from pydantic import ConfigDict

from qutip import Qobj

from oqd_compiler_infrastructure import TypeReflectBaseModel

########################################################################################


class QutipExperiment(TypeReflectBaseModel):
    """
    Class representing a qutip experiment represented in terms of atomic operations expressed in terms of their Hamiltonians.

    Attributes:
        base (Operator): Free Hamiltonian.
        sequence (List[AtomicEmulatorGate]): List of gates to apply.

    """

    model_config = ConfigDict(validate_assignments=True, arbitrary_types_allowed=True)

    base: Union[Qobj, Callable[[float], Qobj]]
    sequence: List[QutipGate]


class QutipGate(TypeReflectBaseModel):
    """
    Class representing a qutip gate represented in terms of atomic operations expressed in terms of their Hamiltonians.

    Attributes:
        hamiltonian (Operator): Hamiltonian to evolve by.
        duration (float): Time to evolve for.
    """

    model_config = ConfigDict(validate_assignments=True, arbitrary_types_allowed=True)

    hamiltonian: Union[Qobj, Callable[[float], Qobj]]
    duration: float
