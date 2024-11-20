import numpy as np

from qutip import tensor, basis, sesolve

from oqd_compiler_infrastructure import RewriteRule

########################################################################################


class QutipVM(RewriteRule):
    def __init__(self, hilbert_space, fock_cutoff):

        self.hilbert_space = [fock_cutoff if h == "f" else h for h in hilbert_space]

        self.result = None
        pass

    def map_QutipExperiment(self, model):

        initial_state = tensor([basis(h, 0) for h in self.hilbert_space])

        tspan = np.arange(0, model.duration, 1e-3)
        self.result = sesolve(model.hamiltonian, initial_state, tspan)
