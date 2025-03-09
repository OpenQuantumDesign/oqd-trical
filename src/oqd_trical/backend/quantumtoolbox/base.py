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

import os
import subprocess

import numpy as np
from oqd_compiler_infrastructure import Chain, Post
from oqd_core.backend.base import BackendBase
from oqd_core.interface.atomic import AtomicCircuit

from oqd_trical.backend.quantumtoolbox.codegen import QuantumToolboxCodeGeneration
from oqd_trical.light_matter.compiler.analysis import GetHilbertSpace, HilbertSpace
from oqd_trical.light_matter.compiler.canonicalize import (
    RelabelStates,
    canonicalize_emulator_circuit_factory,
)
from oqd_trical.light_matter.compiler.codegen import ConstructHamiltonian
from oqd_trical.light_matter.interface.emulator import AtomicEmulatorCircuit

########################################################################################


class QuantumToolboxBackend(BackendBase):
    """Backend for running simulation of AtomicCircuit with QuantumToolbox.jl

    Attributes:
        save_intermediate (bool): Whether compiler saves the intermediate representation of the atomic circuit
        approx_pass (PassBase): Pass of approximations to apply to the system.
        solver (Literal["SESolver","MESolver"]): QuantumToolbox solver to use.
        solver_options (Dict[str,Any]): QuantumToolbox solver options
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

    def compile(self, circuit, fock_cutoff):
        """
        Compiles a AtomicCircuit or AtomicEmulatorCircuit to a [`QuantumToolboxExperiment`][oqd_trical.backend.quantumtoolbox.interface.QuantumToolboxExperiment].

        Args:
            circuit (Union[AtomicCircuit,AtomicEmulatorCircuit]): circuit to be compiled.
            fock_cutoff (int): Truncation for fock spaces.

        Returns:
            experiment (QuantumToolboxExperiment): Compiled [`QuantumToolboxExperiment`][oqd_trical.backend.quantumtoolbox.interface.QuantumToolboxExperiment].
            hilbert_space (Dict[str, int]): Hilbert space of the system.
        """
        assert isinstance(circuit, (AtomicCircuit, AtomicEmulatorCircuit))

        if isinstance(circuit, AtomicCircuit):
            conversion = Post(ConstructHamiltonian())
            intermediate = conversion(circuit)
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

        hilbert_space = get_hilbert_space.hilbert_space
        _hilbert_space = hilbert_space.hilbert_space
        for k in _hilbert_space.keys():
            if k[0] == "P":
                _hilbert_space[k] = set(range(fock_cutoff))
        hilbert_space = HilbertSpace(hilbert_space=_hilbert_space)

        relabeller = Post(RelabelStates(hilbert_space.get_relabel_rules()))
        intermediate = relabeller(intermediate)

        if self.save_intermediate:
            self.intermediate = intermediate

        compiler_p3 = Post(QuantumToolboxCodeGeneration(hilbert_space=hilbert_space))
        experiment = compiler_p3(intermediate)

        return experiment, hilbert_space

    def run(
        self,
        experiment,
        hilbert_space,
        *,
        julia_options=dict(home=None, project=None, sysimage=None),
    ):
        """
        Runs a [`QuantumToolboxExperiment`][oqd_trical.backend.quantumtoolbox.interface.QuantumToolboxExperiment].

        Args:
            experiment (QuantumToolboxExperiment): [`QuantumToolboxExperiment`][oqd_trical.backend.quantumtoolbox.interface.QuantumToolboxExperiment] to be executed.
            hilbert_space (Dict[str, int]): Hilbert space of the system.
            timestep (float): Timestep between tracked states of the evolution.

        Returns:
            result (Dict[str,Any]): Result of execution of [`QuantumToolboxExperiment`][oqd_trical.backend.quantumtoolbox.interface.QuantumToolboxExperiment].
        """

        with open("__experiment.jl", "w") as f:
            f.write(experiment)

        command = ["julia"]
        if "home" in julia_options.keys() and julia_options["home"]:
            command.append("--home", julia_options["home"])
        if "project" in julia_options.keys() and julia_options["project"]:
            command.append(f"--project={julia_options['project']}")
        if "sysimage" in julia_options.keys() and julia_options["sysimage"]:
            command.extend(["--sysimage", julia_options["sysimage"]])
        command.append("__experiment.jl")

        p = subprocess.run(args=command, capture_output=True, text=True)

        if p.stdout:
            header = "{:=^100}".format(" stdout ")
            print(f"{header}\n{p.stdout}")
        if p.stderr:
            header = "{:=^100}".format(" stderr ")
            print(f"{header}\n{p.stderr}")

        tspan = np.load("__times.npz")
        states = np.load("__states.npz")

        os.remove("__experiment.jl")
        os.remove("__times.npz")
        os.remove("__states.npz")

        return dict(
            tspan=tspan,
            states=states,
            final_state=states[-1],
        )
