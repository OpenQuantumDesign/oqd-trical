from __future__ import annotations
from typing import List, Optional

from oqd_compiler_infrastructure import TypeReflectBaseModel

########################################################################################

from .operator import OperatorSubTypes

########################################################################################


class AtomicEmulatorCircuit(TypeReflectBaseModel):
    """
    Class representing a quantum information experiment represented in terms of atomic operations expressed in terms of their Hamiltonians.

    Attributes:
        frame (Optional[Operator]): Operator that defines the rotating frame of reference.
        base (Operator): Free Hamiltonian.
        sequence (List[AtomicEmulatorGate]): List of gates to apply.

    """

    frame: Optional[OperatorSubTypes] = None
    base: OperatorSubTypes
    sequence: List[AtomicEmulatorGate]


class AtomicEmulatorGate(TypeReflectBaseModel):
    """
    Class representing a gate represented in terms of atomic operations expressed in terms of their Hamiltonians.

    Attributes:
        hamiltonian (Operator): Hamiltonian to evolve by.
        duration (float): Time to evolve for.
    """

    hamiltonian: OperatorSubTypes
    duration: float
