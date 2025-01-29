from oqd_core.backend.base import BackendBase

from oqd_compiler_infrastructure import In, Chain, Post, Pre

########################################################################################

from ...light_matter.analysis import AnalyseHilbertSpace
from ...light_matter.codegen import ConstructHamiltonian
from ...light_matter.canonicalize import canonicalization_pass_factory

from .codegen import QutipCodeGeneration
from .vm import QutipVM

########################################################################################


class QutipBackend(BackendBase):
    def __init__(self, save_intermediate=True):
        super().__init__()

        self.save_intermediate = save_intermediate
        self.intermediate = None

    def compile(self, circuit, fock_cutoff):
        analysis = In(AnalyseHilbertSpace())

        analysis(circuit)

        hilbert_space = analysis.children[0].hilbert_space

        for k in hilbert_space.keys():
            if k[0] == "P":
                hilbert_space[k] = fock_cutoff

        compiler_p1 = Chain(
            Post(ConstructHamiltonian()),
            canonicalization_pass_factory(),
        )

        compiler_p2 = Post(QutipCodeGeneration(hilbert_space=hilbert_space))

        intermediate = compiler_p1(circuit)

        if self.save_intermediate:
            self.intermediate = intermediate

        experiment = compiler_p2(intermediate)

        return experiment, hilbert_space

    def run(self, experiment, hilbert_space, timestep):
        vm = Pre(
            QutipVM(
                hilbert_space=hilbert_space,
                timestep=timestep,
            )
        )

        vm(experiment)

        return vm.children[0].result
