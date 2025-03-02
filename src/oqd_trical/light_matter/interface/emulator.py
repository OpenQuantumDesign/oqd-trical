# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        frame (Optional[Operator]): [`Operator`][oqd_trical.light_matter.interface.operator.Operator] that defines the rotating frame of reference.
        base (Operator): Free Hamiltonian.
        sequence (List[AtomicEmulatorGate]): List of gates to apply.

    """

    frame: Optional[OperatorSubTypes] = None
    sequence: List[AtomicEmulatorGate]


class AtomicEmulatorGate(TypeReflectBaseModel):
    """
    Class representing a gate represented in terms of atomic operations expressed in terms of their Hamiltonians.

    Attributes:
        hamiltonian (Operator): Hamiltonian to evolve by.
        duration (float): Time to evolve for.
    """

    hamiltonian: OperatorSubTypes
    dissipation: List[OperatorSubTypes] = []
    duration: float
