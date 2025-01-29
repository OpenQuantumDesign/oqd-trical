from __future__ import annotations
from typing import Union, List

from oqd_compiler_infrastructure import TypeReflectBaseModel
from oqd_core.interface.math import CastMathExpr

########################################################################################

from .operator import OperatorSubTypes

########################################################################################


class AtomicEmulatorCircuit(TypeReflectBaseModel):
    """Class representing a quantum information experiment represented in terms of atomic operations expressed in terms of their Hamiltonians."""

    base: OperatorSubTypes
    sequence: List[AtomicEmulatorGate]


class AtomicEmulatorGate(TypeReflectBaseModel):
    """Class representing a gate represented in terms of atomic operations expressed in terms of their Hamiltonians."""

    hamiltonian: OperatorSubTypes
    duration: float
