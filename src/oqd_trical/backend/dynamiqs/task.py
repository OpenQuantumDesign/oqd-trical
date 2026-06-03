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

"""Task-level arguments for atomic-layer simulation with the Dynamiqs backend."""

from __future__ import annotations

from typing import Optional

from oqd_core.backend.task import TaskArgsAtomic
from pydantic import BaseModel, ConfigDict, Field

from oqd_trical.backend.dynamiqs.solver import DynamiqsSolverOptions

########################################################################################


class TaskArgsAtomicEmulator(BaseModel):
    """Extends atomic task args with optional Dynamiqs/Diffrax settings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fock_trunc: int = Field(default=4, description="Fock space truncation per phonon mode.")
    dt: float = Field(
        default=1e-8,
        description="Output timestep between saved states (seconds).",
    )
    dynamiqs_solver_options: Optional[DynamiqsSolverOptions] = Field(
        default=None,
        description="Diffrax integrator settings (Tsit5, rtol/atol, time rescaling).",
    )


def task_args_from_atomic(
    args: TaskArgsAtomic | TaskArgsAtomicEmulator,
) -> tuple[int, float, Optional[DynamiqsSolverOptions]]:
    """Normalize task args into (fock_trunc, dt, dynamiqs_solver_options)."""
    fock = args.fock_trunc
    dt = args.dt
    solver = (
        args.dynamiqs_solver_options
        if isinstance(args, TaskArgsAtomicEmulator)
        else None
    )
    return fock, dt, solver
