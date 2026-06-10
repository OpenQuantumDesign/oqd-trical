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

from typing import Literal, Optional

from oqd_core.backend.task import TaskArgsAtomic

from oqd_trical.backend.dynamiqs.solver_options import DynamiqsSolverOptions

########################################################################################


class TaskArgsAtomicEmulator(TaskArgsAtomic):
    """Task arguments for the Dynamiqs backend.

    Extends TaskArgsAtomic with the Diffrax solver and its options, so a Task can
    carry them to
    [`DynamiqsBackend.run_task`][oqd_trical.backend.dynamiqs.DynamiqsBackend.run_task].
    The inherited fock_trunc and dt set the Fock cutoff and the timestep.

    Attributes:
        solver (Literal["SESolver","MESolver"]): Dynamiqs solver to use.
        solver_options (Optional[DynamiqsSolverOptions]): Dynamiqs solver options.
    """

    solver: Literal["SESolver", "MESolver"] = "SESolver"
    solver_options: Optional[DynamiqsSolverOptions] = None
