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

from oqd_compiler_infrastructure import Chain, Post, Pre
from oqd_core.backend.base import BackendBase
from oqd_core.compiler.atomic.canonicalize import canonicalize_atomic_circuit_factory
from oqd_core.interface.atomic import AtomicCircuit

from oqd_trical.backend.dynamiqs.codegen import DynamiqsCodeGeneration
from oqd_trical.backend.dynamiqs.vm import DynamiqsVM
from oqd_trical.light_matter.compiler.analysis import GetHilbertSpace, HilbertSpace
from oqd_trical.light_matter.compiler.canonicalize import (
    RelabelStates,
    canonicalize_emulator_circuit_factory,
)
from oqd_trical.light_matter.compiler.codegen import ConstructHamiltonian
from oqd_trical.light_matter.interface.emulator import AtomicEmulatorCircuit

########################################################################################


class DynamiqsBackend(BackendBase):
    """Backend for running simulation of AtomicCircuit with Dynamiqs

    Attributes:
        save_intermediate (bool): Whether compiler saves the intermediate representation of the atomic circuit
        approx_pass (PassBase): Pass of approximations to apply to the system.
        solver (Literal["SESolver","MESolver"]): Dynamiqs solver to use.
        solver_options (Dict[str,Any]): Dynamiqs solver options
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

    def compile(self, circuit, fock_cutoff, *, relabel=True):
        """
        Compiles a AtomicCircuit or AtomicEmulatorCircuit to a [`DynamiqsExperiment`][oqd_trical.backend.dynamiqs.interface.DynamiqsExperiment].

        Args:
            circuit (Union[AtomicCircuit,AtomicEmulatorCircuit]): circuit to be compiled.
            fock_cutoff (int, Dict[str, int]): Truncation for fock spaces.

        Returns:
            experiment (DynamiqsExperiment): Compiled [`DynamiqsExperiment`][oqd_trical.backend.dynamiqs.interface.DynamiqsExperiment].
            hilbert_space (Dict[str, int]): Hilbert space of the system.
        """
        assert isinstance(circuit, (AtomicCircuit, AtomicEmulatorCircuit))

        if isinstance(circuit, AtomicCircuit):
            canonicalize = canonicalize_atomic_circuit_factory()
            intermediate = canonicalize(circuit)
            conversion = Post(ConstructHamiltonian())
            intermediate = conversion(intermediate)
        else:
            intermediate = circuit

        intermediate = canonicalize_emulator_circuit_factory()(intermediate)

        if self.approx_pass:
            intermediate = Chain(
                self.approx_pass, canonicalize_emulator_circuit_factory()
            )(intermediate)

        get_hilbert_space = GetHilbertSpace()
        analysis = Post(get_hilbert_space)
        analysis(intermediate)

        if relabel:
            analysis(intermediate)
        else:
            analysis(circuit.system)

        hilbert_space = get_hilbert_space.hilbert_space
        _hilbert_space = hilbert_space.hilbert_space
        for k in _hilbert_space.keys():
            if k[0] == "P":
                if isinstance(fock_cutoff, int):
                    _hilbert_space[k] = set(range(fock_cutoff))
                else:
                    _hilbert_space[k] = set(range(fock_cutoff[k]))
        hilbert_space = HilbertSpace(hilbert_space=_hilbert_space)

        if any(map(lambda x: x is None, hilbert_space.hilbert_space.values())):
            raise "Hilbert space not fully specified."

        relabeller = Post(RelabelStates(hilbert_space.get_relabel_rules()))
        intermediate = relabeller(intermediate)

        if self.save_intermediate:
            self.intermediate = intermediate

        compiler_p3 = Post(DynamiqsCodeGeneration(hilbert_space=hilbert_space))
        experiment = compiler_p3(intermediate)

        return experiment, hilbert_space

    def run(self, experiment, hilbert_space, timestep, *, initial_state=None):
        """
        Runs a [`DynamiqsExperiment`][oqd_trical.backend.dynamiqs.interface.DynamiqsExperiment].

        Args:
            experiment (DynamiqsExperiment): [`DynamiqsExperiment`][oqd_trical.backend.dynamiqs.interface.DynamiqsExperiment] to be executed.
            hilbert_space (Dict[str, int]): Hilbert space of the system.
            timestep (float): Timestep between tracked states of the evolution.

        Returns:
            result (Dict[str,Any]): Result of execution of [`DynamiqsExperiment`][oqd_trical.backend.dynamiqs.interface.DynamiqsExperiment].
        """
        vm = Pre(
            DynamiqsVM(
                hilbert_space=hilbert_space,
                timestep=timestep,
                solver=self.solver,
                solver_options=self.solver_options,
                initial_state=initial_state,
            )
        )

        vm(experiment)

        return vm.children[0].result
