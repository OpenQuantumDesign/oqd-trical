from __future__ import annotations

from typing import List

from pydantic import ConfigDict

from qutip import Qobj

from oqd_compiler_infrastructure import TypeReflectBaseModel

########################################################################################


class QutipExperiment(TypeReflectBaseModel):
    model_config = ConfigDict(validate_assignments=True, arbitrary_types_allowed=True)

    base: Qobj
    sequence: List[QutipGate]


class QutipGate(TypeReflectBaseModel):
    model_config = ConfigDict(validate_assignments=True, arbitrary_types_allowed=True)

    hamiltonian: Qobj
    duration: float
