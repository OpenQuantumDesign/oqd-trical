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

########################################################################################

import numpy as np
import pytest

try:
    import dynamiqs as dq
    import qutip as qt
except ImportError:
    pytest.skip("dynamiqs or qutip not installed", allow_module_level=True)

from oqd_compiler_infrastructure import Chain, Post
from oqd_core.backend.task import Task
from oqd_core.interface.atomic import (
    AtomicCircuit,
    Beam,
    Phonon,
    Pulse,
    SequentialProtocol,
    System,
    Yb171IIBuilder,
)

from oqd_trical.backend import DynamiqsBackend, DynamiqsSolverOptions, QutipBackend
from oqd_trical.backend.dynamiqs.task import TaskArgsAtomicEmulator
from oqd_trical.backend.dynamiqs.vm import DynamiqsVM
from oqd_trical.backend.qutip.vm import QutipVM
from oqd_trical.light_matter.compiler.analysis import HilbertSpace
from oqd_trical.light_matter.compiler.approximate import (
    RotatingReferenceFrame,
    RotatingWaveApprox,
)
from oqd_trical.light_matter.compiler.canonicalize import (
    canonicalize_emulator_circuit_factory,
)

########################################################################################

# Population tolerance for QuTiP vs Dynamiqs agreement on matched Hamiltonian units.
POPULATION_RTOL = 5e-2
TIMESTEP = 1e-8
FOCK_CUTOFF = 3


def _approx_pass(frame_specs):
    return Chain(
        Post(RotatingReferenceFrame(frame_specs=frame_specs)),
        canonicalize_emulator_circuit_factory(),
        Post(RotatingWaveApprox(cutoff=2 * np.pi * 1e9)),
    )


def _excited_populations(result, hilbert_space, backend):
    n = hilbert_space.size["E0"]
    nf = hilbert_space.size["P0"]
    if backend == "qutip":
        proj = qt.tensor(
            qt.basis(n, 1) * qt.basis(n, 1).dag(),
            qt.qeye(nf),
        )
        return [float(qt.expect(proj, state).real) for state in result["states"]]
    proj = dq.tensor(dq.basis(n, 1) @ dq.basis(n, 1).dag(), dq.eye(nf))
    return [float(dq.expect(proj, state).real) for state in result["states"]]


def _run_both(circuit, frame_specs):
    approx = _approx_pass(frame_specs)
    q_backend = QutipBackend(approx_pass=approx, solver_options={"progress_bar": False})
    d_backend = DynamiqsBackend(approx_pass=approx)
    q_exp, hs = q_backend.compile(circuit, FOCK_CUTOFF)
    d_exp, _ = d_backend.compile(circuit, FOCK_CUTOFF)
    q_res = q_backend.run(q_exp, hs, TIMESTEP)
    d_res = d_backend.run(d_exp, hs, TIMESTEP)
    return q_res, d_res, hs


def _microwave_rabi_circuit():
    ion = Yb171IIBuilder().build(["q0", "q1"])
    com = Phonon(energy=2 * np.pi * 1e6, eigenvector=[1, 0, 0])
    system = System(ions=[ion], modes=[com])
    beam = Beam(
        transition="q0->q1",
        rabi=2 * np.pi * 1e6,
        detuning=0,
        phase=0,
        polarization=[0, 1, 0],
        wavevector=[1, 0, 0],
        target=0,
    )
    protocol = SequentialProtocol(sequence=[Pulse(beam=beam, duration=1e-6)])
    circuit = AtomicCircuit(system=system, protocol=protocol)
    frame_specs = {"E0": [level.energy for level in ion.levels], "P0": 2 * np.pi * 1e6}
    return circuit, frame_specs


def _red_sideband_circuit():
    ion = Yb171IIBuilder().build(["q1", "e0"])
    com = Phonon(energy=2 * np.pi * 1e6, eigenvector=[1, 0, 0])
    system = System(ions=[ion], modes=[com])
    beam = Beam(
        transition="q1->e0",
        rabi=2 * np.pi * 1e6,
        detuning=-2 * np.pi * 1.1e6,
        phase=0,
        polarization=[0, 0, 1],
        wavevector=[1, 0, 0],
        target=0,
    )
    protocol = SequentialProtocol(sequence=[Pulse(beam=beam, duration=1e-5)])
    circuit = AtomicCircuit(system=system, protocol=protocol)
    frame_specs = {"E0": [level.energy for level in ion.levels], "P0": 2 * np.pi * 1e6}
    return circuit, frame_specs


########################################################################################


class TestInitialStateVM:
    def test_qutip_pass(self):
        hilbert_space = HilbertSpace(hilbert_space=dict(E0={0, 1}, E1={0, 1}))
        initial_state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))

        QutipVM(hilbert_space=hilbert_space, timestep=1, initial_state=initial_state)

    @pytest.mark.xfail
    def test_qutip_fail(self):
        hilbert_space = HilbertSpace(hilbert_space=dict(E0={0, 1}, E1={0, 1}))
        initial_state = qt.tensor(qt.basis(2, 0), qt.basis(3, 0))

        QutipVM(hilbert_space=hilbert_space, timestep=1, initial_state=initial_state)

    def test_dynamiqs_pass(self):
        hilbert_space = HilbertSpace(hilbert_space=dict(E0={0, 1}, E1={0, 1}))
        initial_state = dq.tensor(dq.basis(2, 0), dq.basis(2, 0))

        DynamiqsVM(hilbert_space=hilbert_space, timestep=1, initial_state=initial_state)

    @pytest.mark.xfail
    def test_dynamiqs_fail(self):
        hilbert_space = HilbertSpace(hilbert_space=dict(E0={0, 1}, E1={0, 1}))
        initial_state = dq.tensor(dq.basis(2, 0), dq.basis(3, 0))

        DynamiqsVM(hilbert_space=hilbert_space, timestep=1, initial_state=initial_state)


class TestDynamiqsSolverOptions:
    def test_default_progress_meter_disabled(self):
        opts = DynamiqsSolverOptions().to_solver_dict()
        assert opts["options"].progress_meter is False

    def test_backend_accepts_solver_options_model(self):
        backend = DynamiqsBackend(solver_options=DynamiqsSolverOptions(rtol=1e-4))
        assert backend.solver_options["method"].rtol == 1e-4


class TestBackendParity:
    def test_single_ion_rabi_carrier(self):
        circuit, frame_specs = _microwave_rabi_circuit()
        q_res, d_res, hs = _run_both(circuit, frame_specs)
        pq = _excited_populations(q_res, hs, "qutip")
        pd = _excited_populations(d_res, hs, "dynamiqs")
        assert len(pq) == len(pd)
        np.testing.assert_allclose(pq, pd, rtol=POPULATION_RTOL, atol=POPULATION_RTOL)

    def test_single_ion_red_sideband(self):
        circuit, frame_specs = _red_sideband_circuit()
        q_res, d_res, hs = _run_both(circuit, frame_specs)
        pq = _excited_populations(q_res, hs, "qutip")
        pd = _excited_populations(d_res, hs, "dynamiqs")
        assert len(pq) == len(pd)
        np.testing.assert_allclose(pq, pd, rtol=POPULATION_RTOL, atol=POPULATION_RTOL)

    def test_hamiltonian_matches_qutip_at_gate_times(self):
        circuit, frame_specs = _microwave_rabi_circuit()
        approx = _approx_pass(frame_specs)
        q_exp, _ = QutipBackend(approx_pass=approx).compile(circuit, FOCK_CUTOFF)
        d_exp, _ = DynamiqsBackend(approx_pass=approx).compile(circuit, FOCK_CUTOFF)
        Hq = q_exp.sequence[0].hamiltonian
        Hd = d_exp.sequence[0].hamiltonian
        for t in (0.0, 0.5e-6, 1e-6):
            Mq = Hq(t).full()
            Md = np.asarray(Hd(t).to_jax())
            rel = np.linalg.norm(Mq - Md) / np.linalg.norm(Mq)
            assert rel < 1e-4, f"relative Hamiltonian mismatch at t={t}: {rel}"

    def test_run_task_entry_point(self):
        from oqd_core.backend.task import TaskArgsAtomic

        circuit, frame_specs = _microwave_rabi_circuit()
        task = Task(
            program=circuit,
            args=TaskArgsAtomic(fock_trunc=FOCK_CUTOFF, dt=TIMESTEP),
        )
        result = DynamiqsBackend(approx_pass=_approx_pass(frame_specs)).run_task(task)
        assert len(result["states"]) > 1
        assert result["tspan"][-1] == pytest.approx(1e-6)

    def test_run_task_with_emulator_solver_options(self):
        circuit, frame_specs = _microwave_rabi_circuit()
        # oqd-core Task.args is only TaskArgsAtomic today; construct bypasses that union.
        task = Task.model_construct(
            program=circuit,
            args=TaskArgsAtomicEmulator(
                fock_trunc=FOCK_CUTOFF,
                dt=TIMESTEP,
                dynamiqs_solver_options=DynamiqsSolverOptions(rtol=1e-4),
            ),
        )
        backend = DynamiqsBackend(approx_pass=_approx_pass(frame_specs))
        backend.run_task(task)
        assert backend.solver_options["method"].rtol == 1e-4
