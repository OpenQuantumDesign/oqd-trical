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

import numpy as np

from qutip import tensor, basis, SESolver, MESolver, QobjEvo

from oqd_compiler_infrastructure import RewriteRule

########################################################################################


class QutipVM(RewriteRule):
    """
    Rule that executes a [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment].

    Attributes:
        hilbert_space (Dict[str, int]): Hilbert space of the system.
        timestep (float): Timestep between tracked states of the evolution.
        solver (Literal["SESolver","MESolver"]): QuTiP solver to use.
        solver_options (Dict[str,Any]): Qutip solver options
    """

    def __init__(self, hilbert_space, timestep, solver="SESolver", solver_options={}):
        self.hilbert_space = hilbert_space
        self.timestep = timestep

        self.states = []
        self.tspan = []

        self.solver = {
            "SESolver": SESolver,
            "MESolver": MESolver,
        }[solver]
        self.solver_options = solver_options
        pass

    @property
    def result(self):
        return dict(
            final_state=self.current_state, states=self.states, tspan=self.tspan
        )

    def map_QutipExperiment(self, model):
        self.base = model.base

        self.current_state = tensor(
            [basis(self.hilbert_space[k], 0) for k in self.hilbert_space.keys()]
        )

        self.states.append(self.current_state)
        self.tspan.append(0)

    def map_QutipGate(self, model):
        tspan = np.arange(0, model.duration, self.timestep)

        H = QobjEvo(lambda t: model.hamiltonian(t) + self.base(t))
        solver = self.solver(H, options=self.solver_options)

        res = solver.run(
            self.current_state,
            tspan,
        )

        self.current_state = res.final_state

        self.states.extend(list(res.states[1:]))
        self.tspan.extend(list(tspan[1:] + self.tspan[-1]))
