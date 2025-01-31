from oqd_core.backend.base import BackendBase

from oqd_compiler_infrastructure import In, Chain, Post, Pre

########################################################################################

from ...light_matter.analysis import AnalyseHilbertSpace
from ...light_matter.codegen import ConstructHamiltonian
from ...light_matter.canonicalize import canonicalization_pass_factory
from ...light_matter.interface import AtomicEmulatorCircuit

from .codegen import QutipCodeGeneration
from .vm import QutipVM

########################################################################################


class QutipBackend(BackendBase):
    """Backend for running simulation of AtomicCircuit with QuTiP

    Attributes:
        save_intermediate (bool): Whether compiler saves the intermediate representation of the atomic circuit
        intermediate (AtomicEmulatorCircuit): Intermediate representation of the atomic circuit during compilation
    """

    def __init__(
        self,
        save_intermediate=True,
        approx_pass=None,
        solver="SESolver",
        solver_options={},
    ):
        super().__init__()

        self.save_intermediate = save_intermediate
        self.intermediate = None
        self.approx_pass = approx_pass
        self.solver = solver
        self.solver_options = solver_options

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

        intermediate = compiler_p1(circuit)

        if self.approx_pass:
            intermediate = Chain(self.approx_pass, canonicalization_pass_factory())(
                intermediate
            )

        if self.save_intermediate:
            self.intermediate = intermediate

        compiler_p2 = Post(QutipCodeGeneration(hilbert_space=hilbert_space))

        experiment = compiler_p2(intermediate)

        return experiment, hilbert_space

    def run(self, experiment, hilbert_space, timestep):
        vm = Pre(
            QutipVM(
                hilbert_space=hilbert_space,
                timestep=timestep,
                solver=self.solver,
                solver_options=self.solver_options,
            )
        )

        vm(experiment)

        return vm.children[0].result
