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

from typing import Literal

import dynamiqs as dq
from pydantic import BaseModel, ConfigDict

########################################################################################


class DynamiqsSolverOptions(BaseModel):
    """Explicit Diffrax solver options for the Dynamiqs backend.

    Builds the Dynamiqs method and options passed to each sesolve call. Tolerances
    live on the integration method, from which Dynamiqs configures the Diffrax
    PIDController.

    Attributes:
        method (str): Adaptive ODE method, one of Tsit5, Dopri5, Dopri8, Kvaerno3
            or Kvaerno5. Defaults to Tsit5.
        rtol (float): Relative tolerance of the step-size controller. Defaults to 1e-8.
        atol (float): Absolute tolerance of the step-size controller. Defaults to 1e-8.
        max_steps (int): Maximum number of solver steps. Defaults to 1000000.
        progress_meter (bool): Show the Dynamiqs progress meter. Defaults to False to
            avoid a tqdm/ZMQError under Jupyter (issue #26).
    """

    model_config = ConfigDict(frozen=True)

    method: Literal["Tsit5", "Dopri5", "Dopri8", "Kvaerno3", "Kvaerno5"] = "Tsit5"
    rtol: float = 1e-8
    atol: float = 1e-8
    max_steps: int = 1_000_000
    progress_meter: bool = False

    def to_method(self):
        """Build the Dynamiqs adaptive integration method instance."""
        return getattr(dq.method, self.method)(
            rtol=self.rtol, atol=self.atol, max_steps=self.max_steps
        )

    def to_options(self):
        """Build the Dynamiqs solver options."""
        return dq.Options(progress_meter=self.progress_meter)

    @classmethod
    def from_obj(cls, obj):
        """Normalize None, a dict or a DynamiqsSolverOptions into an instance."""
        if obj is None:
            return cls()
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls(**obj)
        raise TypeError(
            "solver_options must be None, a dict, or a DynamiqsSolverOptions, "
            f"got {type(obj).__name__}"
        )
