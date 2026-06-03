# Copyright 2024-2025 Open Quantum Design
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from oqd_compiler_infrastructure import Chain, Post, Pre
from oqd_core.backend.base import BackendBase
from oqd_core.compiler.atomic.canonicalize import canonicalize_atomic_circuit_factory
from oqd_core.interface.atomic import AtomicCircuit

from oqd_trical.backend.qutip.codegen import QutipCodeGeneration
from oqd_trical.backend.qutip.datastore import build_emulator_datastore
from oqd_trical.backend.qutip.vm import QutipVM
from oqd_trical.light_matter.compiler.analysis import GetHilbertSpace, HilbertSpace
from oqd_trical.light_matter.compiler.canonicalize import (
    RelabelStates,
    canonicalize_emulator_circuit_factory,
)
from oqd_trical.light_matter.compiler.codegen import ConstructHamiltonian
from oqd_trical.light_matter.interface.emulator import AtomicEmulatorCircuit

try:
    __version__ = _pkg_version("oqd-trical")
except PackageNotFoundError:
    __version__ = "0+unknown"

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
        solver_options={"progress_bar": True},
    ):
        super().__init__()

        self.save_intermediate = save_intermediate
        self.intermediate = None
        self.approx_pass = approx_pass
        self.solver = solver
        self.solver_options = solver_options

        # Most recently used fock cutoff, populated by `compile` and used by
        # `run` to populate the Datastore attrs. May be overridden by
        # passing `fock_cutoff` to `run`.
        self._fock_cutoff = None

    def compile(self, circuit, fock_cutoff, *, relabel=True):
        """
        Compiles a AtomicCircuit or AtomicEmulatorCircuit to a [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment].

        Args:
            circuit (Union[AtomicCircuit,AtomicEmulatorCircuit]): circuit to be compiled.
            fock_cutoff (int): Truncation for fock spaces.

        Returns:
            experiment (QutipExperiment): Compiled [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment].
            hilbert_space (HilbertSpace): Hilbert space of the system.
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

        compiler_p3 = Post(QutipCodeGeneration(hilbert_space=hilbert_space))
        experiment = compiler_p3(intermediate)

        # Stash for the upcoming `run` call.
        self._fock_cutoff = fock_cutoff

        return experiment, hilbert_space

    def run(
        self,
        experiment,
        hilbert_space,
        timestep,
        *,
        initial_state=None,
        fock_cutoff=None,
    ):
        """
        Runs a [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment]
        and returns the result as a schema-validated
        [`Datastore`][oqd_dataschema.Datastore].

        Args:
            experiment (QutipExperiment): [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment] to be executed.
            hilbert_space (HilbertSpace): Hilbert space of the system.
            timestep (float): Timestep between tracked states of the evolution.
            initial_state: Optional initial state. If ``None``, every
                subsystem is initialized to its ground state.
            fock_cutoff: Optional override for the Fock cutoff used to
                label the run metadata. Defaults to the value passed to
                the most recent :meth:`compile` call.

        Returns:
            result (Datastore): Schema-validated datastore wrapping a
            single [`TrICalEmulatorDataGroup`][oqd_trical.backend.qutip.datastore.TrICalEmulatorDataGroup]
            named ``"emulation"``. The group holds ``tspan``, ``states``,
            ``final_state`` and (optionally) ``frame`` datasets plus the
            run metadata in ``attrs``. The whole datastore can be saved
            to disk with :meth:`Datastore.model_dump_hdf5` and reloaded
            with :meth:`Datastore.model_validate_hdf5`.
        """
        vm = Pre(
            QutipVM(
                hilbert_space=hilbert_space,
                timestep=timestep,
                solver=self.solver,
                solver_options=self.solver_options,
                initial_state=initial_state,
            )
        )

        vm(experiment)
        run_vm = vm.children[0]

        cutoff = fock_cutoff if fock_cutoff is not None else self._fock_cutoff
        # If the caller never invoked `compile` and did not pass a cutoff,
        # fall back to an empty dict (no Fock modes labelled).
        if cutoff is None:
            cutoff = {}

        return build_emulator_datastore(
            states=run_vm.states,
            tspan=run_vm.tspan,
            final_state=run_vm.current_state,
            frame=getattr(run_vm, "frame", None),
            hilbert_space=hilbert_space,
            solver=run_vm.solver_name,
            timestep=timestep,
            fock_cutoff=cutoff,
            backend="qutip",
            version=__version__,
        )
