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

import dynamiqs as dq
from dynamiqs import mesolve, sesolve
from jax import numpy as jnp
from oqd_compiler_infrastructure import RewriteRule

from oqd_trical.backend.dynamiqs.solver import normalize_solver_options

########################################################################################


class DynamiqsVM(RewriteRule):
    """
    Rule that executes a [`DynamiqsExperiment`][oqd_trical.backend.dynamiqs.interface.DynamiqsExperiment].

    Attributes:
        hilbert_space (Dict[str, int]): Hilbert space of the system.
        timestep (float): Timestep between tracked states of the evolution (seconds).
        solver (Literal["SESolver","MESolver"]): Dynamiqs solver to use.
        solver_options (Dict[str,Any]): Keys ``method`` (dq.method.*) and ``options``
            (dq.Options); see [`DynamiqsSolverOptions`][oqd_trical.backend.dynamiqs.solver.DynamiqsSolverOptions].
    """

    def __init__(
        self,
        hilbert_space,
        timestep,
        *,
        initial_state=None,
        solver="SESolver",
        solver_options=None,
    ):
        self.hilbert_space = hilbert_space
        self.timestep = timestep

        if initial_state:
            if initial_state.dims != tuple(self.hilbert_space.size.values()):
                raise ValueError("Initial state incompatible with Hilbert space")
            self.current_state = initial_state
        else:
            self.current_state = dq.tensor(
                *[
                    dq.basis(self.hilbert_space.size[k], 0)
                    for k in self.hilbert_space.size.keys()
                ]
            )

        self.states = [self.current_state]
        self.tspan = [0.0]

        self.solver = {
            "SESolver": sesolve,
            "MESolver": mesolve,
        }[solver]
        self.solver_options = normalize_solver_options(solver_options)

    @property
    def result(self):
        return dict(
            final_state=self.current_state,
            states=self.states,
            tspan=self.tspan,
            frame=self.frame,
            hilbert_space=self.hilbert_space,
        )

    def map_DynamiqsExperiment(self, model):
        self.frame = model.frame

    def map_DynamiqsGate(self, model):
        tspan = jnp.arange(0, model.duration, self.timestep)

        if tspan[-1] != model.duration:
            tspan = jnp.append(tspan, model.duration)

        tspan = tspan + self.tspan[-1]

        empty_hamiltonian = model.hamiltonian is None

        if empty_hamiltonian:
            self.tspan.extend(list(tspan[1:]))
            self.states.extend([self.current_state] * (len(tspan) - 1))
            return

        hamiltonian, tspan_solve = _prepare_hamiltonian_and_tspan(
            model.hamiltonian,
            tspan,
            model.duration,
            rescale_time=self.solver_options.get("rescale_time", True),
            time_scale=self.solver_options.get("time_scale"),
        )

        res = self.solver(
            hamiltonian,
            self.current_state,
            tspan_solve,
            method=self.solver_options["method"],
            options=self.solver_options["options"],
        )

        self.current_state = res.final_state

        self.tspan.extend(list(tspan[1:]))
        self.states.extend(list(res.states[1:]))


def _hamiltonian_scale(hamiltonian, duration: float) -> float:
    """Characteristic angular frequency (rad/s) for dimensionless time rescaling."""
    omega = float(jnp.max(jnp.abs(hamiltonian(0.0).to_jax())))
    if duration > 0:
        omega = max(
            omega, float(jnp.max(jnp.abs(hamiltonian(duration).to_jax())))
        )
    return max(omega, 1.0)


def _prepare_hamiltonian_and_tspan(
    hamiltonian,
    tspan,
    duration: float,
    *,
    rescale_time: bool,
    time_scale: float | None,
):
    """Optionally rescale to dimensionless time for Diffrax (see solver module docstring)."""
    if not rescale_time:
        return hamiltonian, tspan

    omega = float(
        time_scale
        if time_scale is not None
        else _hamiltonian_scale(hamiltonian, duration)
    )
    # Compose with the inner callable; avoid nesting timecallable(TimeQArray).
    base = hamiltonian.f if hasattr(hamiltonian, "f") else hamiltonian
    hamiltonian = dq.timecallable(lambda tau, f=base, om=omega: f(tau / om) / om)
    return hamiltonian, tspan * omega
