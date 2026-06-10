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
from dynamiqs.progress_meter import NoProgressMeter
from jax import numpy as jnp
from oqd_compiler_infrastructure import RewriteRule

# dynamiqs 0.3.1 named the integration method parameter 'solver'; 0.3.2 renamed it
# to 'method'. Detect which name to use at import time so calls stay compatible.
_METHOD_KWARG = "solver" if hasattr(dq, "solver") else "method"

########################################################################################


class DynamiqsVM(RewriteRule):
    """
    Rule that executes a [`DynamiqsExperiment`][oqd_trical.backend.dynamiqs.interface.DynamiqsExperiment].

    Attributes:
        hilbert_space (Dict[str, int]): Hilbert space of the system.
        timestep (float): Timestep between tracked states of the evolution.
        solver (Literal["SESolver","MESolver"]): Dynamiqs solver to use.
        solver_options (Dict[str,Any]): Dynamiqs solver options. Recognised keys:
            - ``"solver"``: integration method passed to sesolve/mesolve (e.g.
              ``dq.Tsit5()``). Defaults to the dynamiqs built-in default.
            - ``"progress_meter"``: ``False`` / ``True`` / an
              ``AbstractProgressMeter`` instance. Defaults to
              ``NoProgressMeter()`` (no output) to avoid exhausting file
              descriptors when the backend is called from Jupyter/ipykernel.
    """

    def __init__(
        self,
        hilbert_space,
        timestep,
        *,
        initial_state=None,
        solver="SESolver",
        solver_options={},
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
        self.solver_options = solver_options

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
            self.tspan.extend(list(tspan[1:] + self.tspan[-1]))
            self.states.extend([self.current_state] * (len(tspan) - 1))
            return

        solver_kwargs = {}
        if "solver" in self.solver_options:
            solver_kwargs[_METHOD_KWARG] = self.solver_options["solver"]

        options = dq.Options(
            progress_meter=self.solver_options.get("progress_meter", NoProgressMeter())
        )

        res = self.solver(
            model.hamiltonian,
            self.current_state,
            tspan,
            **solver_kwargs,
            options=options,
        )

        self.current_state = res.final_state

        self.tspan.extend(list(tspan[1:]))
        self.states.extend(list(res.states[1:]))
