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
    def __init__(self):
        pass

    def compile(self, circuit, fock_cutoff):
        analysis = In(AnalyseHilbertSpace())

        analysis(circuit)

        hilbert_space = analysis.children[0].hilbert_space

        compiler = Chain(
            Post(ConstructHamiltonian()),
            canonicalization_pass_factory(),
            Post(QutipCodeGeneration(fock_cutoff=fock_cutoff)),
        )

        experiment = compiler(circuit)

        return experiment, hilbert_space, fock_cutoff

    def run(self, experiment, hilbert_space, fock_cutoff, timestep):
        vm = Pre(
            QutipVM(
                hilbert_space=hilbert_space,
                fock_cutoff=fock_cutoff,
                timestep=timestep,
            )
        )

        vm(experiment)

        return vm.children[0].result
