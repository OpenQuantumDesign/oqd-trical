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
