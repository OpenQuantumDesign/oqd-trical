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

"""Diffrax / Dynamiqs integration settings for the TrICal Dynamiqs backend.

Unit convention (matches QuTiP and the atomic-interface lowering):
  - Level energies, phonon energies, Rabi frequencies, and detunings are angular
    frequencies in rad/s (inputs often carry explicit factors of 2*pi).
  - Pulse durations are in seconds.
  - The lowered Hamiltonian H and ``tsave`` passed to ``dq.sesolve`` use the same
    rad/s and second units; Dynamiqs integrates d|psi>/dt = -i H |psi> with hbar=1.

Adaptive step-size controllers are sensitive to |H| and the integration interval.
Trapped-ion circuits after rotating-frame and RWA passes typically have |H| ~ 2*pi
* 1e6 rad/s or smaller; default tolerances are relaxed slightly versus Dynamiqs'
library defaults so stiff sideband dynamics do not exhaust ``max_steps``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import dynamiqs as dq

########################################################################################

# Tsit5 uses a Diffrax PID step-size controller (see dynamiqs.method.Tsit5).
# 1e-6 (library default) can exhaust max_steps on ~MHz-scale trapped-ion dynamics;
# 1e-3 matches QuTiP SESolver on reference circuits to within ~5% on populations.
DEFAULT_RTOL = 1e-3
DEFAULT_ATOL = 1e-3
DEFAULT_MAX_STEPS = 2_000_000


@dataclass
class DynamiqsSolverOptions:
    """Options forwarded to ``dq.sesolve`` / ``dq.mesolve`` (Diffrax under the hood).

    Pass an instance via ``DynamiqsBackend(solver_options=...)`` or a plain dict with
    the same keys. Mirrors the role of QuTiP's ``solver_options`` on
    ``QutipBackend``.
    """

    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL
    max_steps: int = DEFAULT_MAX_STEPS
    method: Optional[Any] = None
    options: Optional[Any] = None
    # Rescale t -> omega*t and H -> H/omega so Diffrax sees O(1) frequencies (rad/s in, seconds in).
    rescale_time: bool = True
    time_scale: Optional[float] = None

    def to_solver_dict(self) -> Dict[str, Any]:
        method = self.method
        if method is None:
            method = dq.method.Tsit5(
                rtol=self.rtol, atol=self.atol, max_steps=self.max_steps
            )
        options = self.options
        if options is None:
            options = dq.Options(progress_meter=False)
        return {
            "method": method,
            "options": options,
            "rescale_time": self.rescale_time,
            "time_scale": self.time_scale,
        }


def normalize_solver_options(
    solver_options: Optional[Union[DynamiqsSolverOptions, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if solver_options is None:
        return DynamiqsSolverOptions().to_solver_dict()
    if isinstance(solver_options, DynamiqsSolverOptions):
        return solver_options.to_solver_dict()
    if "method" not in solver_options:
        opts = DynamiqsSolverOptions(
            rtol=solver_options.get("rtol", DEFAULT_RTOL),
            atol=solver_options.get("atol", DEFAULT_ATOL),
            max_steps=solver_options.get("max_steps", DEFAULT_MAX_STEPS),
            method=solver_options.get("method"),
            options=solver_options.get("options"),
            rescale_time=solver_options.get("rescale_time", True),
            time_scale=solver_options.get("time_scale"),
        )
        merged = opts.to_solver_dict()
        merged.update({k: v for k, v in solver_options.items() if k not in merged})
        return merged
    if "options" not in solver_options:
        solver_options = {
            **solver_options,
            "options": dq.Options(progress_meter=False),
        }
    return solver_options
