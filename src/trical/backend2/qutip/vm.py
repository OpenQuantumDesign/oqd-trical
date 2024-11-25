import numpy as np

from qutip import tensor, basis, sesolve

from oqd_compiler_infrastructure import RewriteRule

########################################################################################


class QutipVM(RewriteRule):
    def __init__(self, hilbert_space, fock_cutoff, timestep):

        self.hilbert_space = [fock_cutoff if h == "f" else h for h in hilbert_space]
        self.timestep = timestep

        self.states = []
        self.tspan = []
        pass

    @property
    def result(self):
        return dict(
            final_state=self.current_state, states=self.states, tspan=self.tspan
        )

    def map_QutipExperiment(self, model):
        self.base = model.base

        self.current_state = tensor([basis(h, 0) for h in self.hilbert_space])

        self.states.append(self.current_state)
        self.tspan.append(0)

    def map_QutipGate(self, model):
        tspan = np.arange(0, model.duration, self.timestep)

        res = sesolve(model.hamiltonian + self.base, self.current_state, tspan)

        self.current_state = res.final_state

        self.states.extend(list(res.states[1:]))
        self.tspan.extend(list(tspan[1:] + self.tspan[-1]))
