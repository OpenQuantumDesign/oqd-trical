from __future__ import annotations
from typing import Union, List

from oqd_compiler_infrastructure import TypeReflectBaseModel
from oqd_core.interface.math import CastMathExpr

########################################################################################

from .operator import OperatorSubTypes

########################################################################################


class AtomicEmulatorCircuit(TypeReflectBaseModel):
    base: OperatorSubTypes
    sequence: List[AtomicEmulatorGate]


class AtomicEmulatorGate(TypeReflectBaseModel):
    hamiltonian: OperatorSubTypes
    duration: float
