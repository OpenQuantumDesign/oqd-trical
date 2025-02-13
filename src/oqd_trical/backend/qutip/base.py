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

from oqd_compiler_infrastructure import Chain, In, Post, Pre
from oqd_core.backend.base import BackendBase
from oqd_core.interface.atomic import AtomicCircuit

########################################################################################
from oqd_trical.light_matter.interface.emulator import AtomicEmulatorCircuit
from oqd_trical.light_matter.compiler.analysis import GetHilbertSpace, HilbertSpace
from oqd_trical.light_matter.compiler.canonicalize import (
    canonicalization_pass_factory,
    RelabelStates,
)
from oqd_trical.light_matter.compiler.codegen import ConstructHamiltonian

from .codegen import QutipCodeGeneration
from .vm import QutipVM

########################################################################################


class QutipBackend(BackendBase):
    """Backend for running simulation of AtomicCircuit with QuTiP

    Attributes:
        save_intermediate (bool): Whether compiler saves the intermediate representation of the atomic circuit
        approx_pass (PassBase): Pass of approximations to apply to the system.
        solver (Literal["SESolver","MESolver"]): QuTiP solver to use.
        solver_options (Dict[str,Any]): Qutip solver options
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
        """
        Compiles a AtomicCircuit or AtomicEmulatorCircuit to a [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment].

        Args:
            circuit (Union[AtomicCircuit,AtomicEmulatorCircuit]): circuit to be compiled.
            fock_cutoff (int): Truncation for fock spaces.

        Returns:
            experiment (QutipExperiment): Compiled [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment].
            hilbert_space (Dict[str, int]): Hilbert space of the system.
        """
        assert isinstance(circuit, (AtomicCircuit, AtomicEmulatorCircuit))

        if isinstance(circuit, AtomicCircuit):
            conversion = Post(ConstructHamiltonian())
            intermediate = conversion(circuit)
        else:
            intermediate = circuit

        intermediate = canonicalization_pass_factory()(circuit)

        if self.approx_pass:
            intermediate = Chain(self.approx_pass, canonicalization_pass_factory())(
                intermediate
            )

        get_hilbert_space = GetHilbertSpace()
        analysis = Post(get_hilbert_space)
        analysis(intermediate)

        hilbert_space = get_hilbert_space.hilbert_space
        _hilbert_space = hilbert_space.hilbert_space
        for k in _hilbert_space.keys():
            if k[0] == "P":
                hilbert_space[k] = set(range(fock_cutoff))
        hilbert_space = HilbertSpace(hilbert_space=_hilbert_space)

        relabeller = Post(RelabelStates(hilbert_space.get_relabel_rules()))
        intermediate = relabeller(intermediate)

        if self.save_intermediate:
            self.intermediate = intermediate

        compiler_p3 = Post(QutipCodeGeneration(hilbert_space=hilbert_space))
        experiment = compiler_p3(intermediate)

        return experiment, hilbert_space

    def run(self, experiment, hilbert_space, timestep):
        """
        Runs a [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment].

        Args:
            experiment (QutipExperiment): [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment] to be executed.
            hilbert_space (Dict[str, int]): Hilbert space of the system.
            timestep (float): Timestep between tracked states of the evolution.

        Returns:
            result (Dict[str,Any]): Result of execution of [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment].
        """
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
