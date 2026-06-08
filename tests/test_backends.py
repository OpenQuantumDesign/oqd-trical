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
import dynamiqs as dq
import jax
import numpy as np
import pytest
import qutip as qt
from jax import numpy as jnp
from oqd_compiler_infrastructure import Chain, Post
from oqd_core.backend.task import Task, TaskArgsAtomic
from oqd_core.interface.atomic import (
    AtomicCircuit,
    Beam,
    Phonon,
    Pulse,
    SequentialProtocol,
    System,
    Yb171IIBuilder,
)

from oqd_trical.backend import (
    DynamiqsBackend,
    DynamiqsSolverOptions,
    QutipBackend,
    TaskArgsAtomicEmulator,
)
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

# Compare against the QuTiP reference in double precision (QuTiP uses float64).
dq.set_precision("double")

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


########################################################################################

FOCK_CUTOFF = 3
TIMESTEP = 1e-8


def _approx_pass(levels):
    frame_specs = {
        "E0": [level.energy for level in levels],
        "P0": 2 * np.pi * 1e6,
    }
    return Chain(
        Post(RotatingReferenceFrame(frame_specs=frame_specs)),
        canonicalize_emulator_circuit_factory(),
        Post(RotatingWaveApprox(cutoff=2 * np.pi * 1e9)),
    )


def _microwave_circuit():
    """Single-ion Rabi flop on the carrier (examples/direct/1_microwave)."""
    ion = Yb171IIBuilder().build(["q0", "q1"])
    system = System(
        ions=[ion], modes=[Phonon(energy=2 * np.pi * 1e6, eigenvector=[1, 0, 0])]
    )
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
    return ion, AtomicCircuit(system=system, protocol=protocol)


def _red_sideband_circuit():
    """Single-ion + COM phonon red sideband (examples/direct/3a_red_sideband)."""
    ion = Yb171IIBuilder().build(["q1", "e0"])
    system = System(
        ions=[ion], modes=[Phonon(energy=2 * np.pi * 1e6, eigenvector=[1, 0, 0])]
    )
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
    return ion, AtomicCircuit(system=system, protocol=protocol)


def _run_qutip(circuit, ion, *, initial_state=None):
    backend = QutipBackend(
        approx_pass=_approx_pass(ion.levels),
        solver_options={"progress_bar": False, "atol": 1e-10, "rtol": 1e-10},
    )
    experiment, hilbert_space = backend.compile(circuit, FOCK_CUTOFF)
    return backend.run(
        experiment,
        hilbert_space=hilbert_space,
        timestep=TIMESTEP,
        initial_state=initial_state,
    )


def _run_dynamiqs(circuit, ion, *, initial_state=None, solver_options=None):
    backend = DynamiqsBackend(
        approx_pass=_approx_pass(ion.levels),
        solver_options=solver_options or DynamiqsSolverOptions(rtol=1e-8, atol=1e-8),
    )
    experiment, hilbert_space = backend.compile(circuit, FOCK_CUTOFF)
    result = backend.run(
        experiment,
        hilbert_space=hilbert_space,
        timestep=TIMESTEP,
        initial_state=initial_state,
    )
    return backend, experiment, hilbert_space, result


def _qutip_populations(states):
    return np.array([np.abs(s.full().flatten()) ** 2 for s in states])


def _dynamiqs_populations(states):
    return np.array([np.abs(np.asarray(s.to_jax()).flatten()) ** 2 for s in states])


def _electronic_populations(pops, dim_e, dim_p):
    return pops.reshape(pops.shape[0], dim_e, dim_p).sum(axis=2)


class TestRabiSidebandParity:
    """Dynamiqs reproduces the QuTiP reference on the example circuits."""

    def test_microwave_rabi_carrier_parity(self):
        ion, circuit = _microwave_circuit()

        rq = _run_qutip(circuit, ion)
        _, _, _, rd = _run_dynamiqs(circuit, ion)

        pq = _qutip_populations(rq["states"])
        pd = _dynamiqs_populations(rd["states"])

        # identical save-time grid and number of saved states across backends
        assert pq.shape == pd.shape
        np.testing.assert_allclose(
            np.array(rq["tspan"]), np.array(rd["tspan"]), atol=1e-15
        )

        # Dynamiqs reproduces QuTiP populations to a small tolerance
        # observed agreement is ~1e-8 in double precision; 1e-5 keeps margin
        np.testing.assert_allclose(pd, pq, atol=1e-5)

        # probability is conserved
        np.testing.assert_allclose(pd.sum(axis=1), 1.0, atol=1e-6)

        # resonant carrier, one full period: invert to q1 at T/2, back to q0 at T
        pe = _electronic_populations(pd, 2, 3)
        assert pe[0, 0] > 0.99
        assert pe[len(pe) // 2, 1] > 0.99
        assert pe[-1, 0] > 0.99
        np.testing.assert_allclose(pe, _electronic_populations(pq, 2, 3), atol=1e-5)

    def test_red_sideband_parity(self):
        ion, circuit = _red_sideband_circuit()
        psi0_q = qt.tensor(qt.basis(2, 0), qt.basis(3, 1))
        psi0_d = dq.tensor(dq.basis(2, 0), dq.basis(3, 1))

        rq = _run_qutip(circuit, ion, initial_state=psi0_q)
        _, _, _, rd = _run_dynamiqs(circuit, ion, initial_state=psi0_d)

        pq = _qutip_populations(rq["states"])
        pd = _dynamiqs_populations(rd["states"])

        assert pq.shape == pd.shape
        # observed agreement is ~1e-8 in double precision; 1e-5 keeps margin
        np.testing.assert_allclose(pd, pq, atol=1e-5)
        np.testing.assert_allclose(pd.sum(axis=1), 1.0, atol=1e-6)

        # sideband transfers population from q1 into e0
        pe = _electronic_populations(pd, 2, 3)
        assert pe[0, 0] > 0.99  # starts in q1
        assert pe[:, 1].max() > 0.1  # excited state e0 becomes populated

    def test_hamiltonian_matches_qutip_at_gate_times(self):
        ion, circuit = _microwave_circuit()
        qb = QutipBackend(
            approx_pass=_approx_pass(ion.levels),
            solver_options={"progress_bar": False},
        )
        db = DynamiqsBackend(approx_pass=_approx_pass(ion.levels))
        qexp, _ = qb.compile(circuit, FOCK_CUTOFF)
        dexp, _ = db.compile(circuit, FOCK_CUTOFF)

        hq = qexp.sequence[0].hamiltonian
        hd = dexp.sequence[0].hamiltonian
        for t in (0.0, 5e-7, 1e-6):
            mq = hq(t).full()
            md = np.asarray(hd(t).to_jax())
            assert np.linalg.norm(md - mq) / max(np.linalg.norm(mq), 1.0) < 1e-6


class TestDynamiqsJit:
    """The Dynamiqs solve path is jax.jit-compatible and differentiable."""

    def test_sesolve_path_is_jit_pure(self):
        ion, circuit = _microwave_circuit()
        backend = DynamiqsBackend(
            approx_pass=_approx_pass(ion.levels),
            solver_options=DynamiqsSolverOptions(rtol=1e-8, atol=1e-8),
        )
        experiment, hilbert_space = backend.compile(circuit, FOCK_CUTOFF)

        hamiltonian = experiment.sequence[0].hamiltonian
        method = backend.solver_options.to_method()
        options = backend.solver_options.to_options()
        tsave = jnp.linspace(0.0, 1e-6, 11)

        @jax.jit
        def evolve(psi0):
            res = dq.sesolve(hamiltonian, psi0, tsave, method=method, options=options)
            return res.final_state

        psi0 = dq.tensor(
            *[dq.basis(hilbert_space.size[k], 0) for k in hilbert_space.size.keys()]
        )

        # compiles + runs with no ConcretizationTypeError -> the solve is jit-pure
        final = evolve(psi0)
        arr = np.asarray(final.to_jax())
        assert np.all(np.isfinite(arr))
        np.testing.assert_allclose((np.abs(arr) ** 2).sum(), 1.0, atol=1e-6)

    def test_sesolve_path_is_differentiable(self):
        ion, circuit = _microwave_circuit()
        backend = DynamiqsBackend(
            approx_pass=_approx_pass(ion.levels),
            solver_options=DynamiqsSolverOptions(rtol=1e-8, atol=1e-8),
        )
        experiment, hilbert_space = backend.compile(circuit, FOCK_CUTOFF)

        hamiltonian = experiment.sequence[0].hamiltonian
        method = backend.solver_options.to_method()
        options = backend.solver_options.to_options()
        tsave = jnp.linspace(0.0, 5e-7, 6)
        phonon = dq.basis(3, 0)

        def excited_population(theta):
            elec = (
                jnp.cos(theta) * dq.basis(2, 0).to_jax()
                + jnp.sin(theta) * dq.basis(2, 1).to_jax()
            )
            psi0 = dq.tensor(dq.asqarray(elec), phonon)
            res = dq.sesolve(
                hamiltonian, psi0, tsave, method=method, options=options
            )
            return jnp.abs(res.final_state.to_jax().flatten()[3:6]).sum()

        # End-to-end autodiff through the compiled Hamiltonian + sesolve.
        grad = jax.grad(excited_population)(0.3)
        assert np.isfinite(grad)
        assert abs(grad) > 1e-3  # genuine dependence, not a zero/constant gradient


class TestDynamiqsSolverOptions:
    """Explicit, overridable Diffrax solver options."""

    def test_defaults(self):
        opts = DynamiqsSolverOptions()
        assert opts.method == "Tsit5"
        assert opts.rtol == 1e-8 and opts.atol == 1e-8
        # progress meter off by default -> avoids the Jupyter ZMQError (#26)
        assert opts.progress_meter is False
        assert opts.to_options().progress_meter is False

    def test_from_obj_normalizes_dict_and_none(self):
        assert DynamiqsSolverOptions.from_obj(None) == DynamiqsSolverOptions()
        assert DynamiqsSolverOptions.from_obj({"rtol": 1e-10}).rtol == 1e-10
        passthrough = DynamiqsSolverOptions(atol=1e-9)
        assert DynamiqsSolverOptions.from_obj(passthrough) is passthrough

    def test_override_reaches_method(self):
        opts = DynamiqsSolverOptions(method="Dopri5", rtol=1e-10, atol=1e-9)
        method = opts.to_method()
        assert method.rtol == 1e-10
        assert method.atol == 1e-9


class TestRunTask:
    """run_task drives the backend from a validated Task (the TaskArgs path)."""

    def test_task_args_emulator_is_a_validated_subclass(self):
        # subclassing TaskArgsAtomic lets it ride the oqd-core Task.args union
        # without Task.model_construct
        _, circuit = _microwave_circuit()
        args = TaskArgsAtomicEmulator(
            fock_trunc=FOCK_CUTOFF,
            dt=TIMESTEP,
            solver_options=DynamiqsSolverOptions(rtol=1e-9),
        )
        assert isinstance(args, TaskArgsAtomic)
        task = Task(program=circuit, args=args)
        assert type(task.args) is TaskArgsAtomicEmulator
        assert task.args.solver_options.rtol == 1e-9

    def test_run_task_runs_the_rabi_flop(self):
        ion, circuit = _microwave_circuit()
        backend = DynamiqsBackend(approx_pass=_approx_pass(ion.levels))
        args = TaskArgsAtomicEmulator(
            fock_trunc=FOCK_CUTOFF,
            dt=TIMESTEP,
            solver_options=DynamiqsSolverOptions(rtol=1e-8, atol=1e-8),
        )
        result = backend.run_task(Task(program=circuit, args=args))

        pe = _electronic_populations(_dynamiqs_populations(result["states"]), 2, 3)
        assert pe[0, 0] > 0.99
        assert pe[len(pe) // 2, 1] > 0.99
        assert pe[-1, 0] > 0.99


class TestEmptyGateTimeline:
    """A pruned (empty) gate advances time monotonically without double-offset."""

    def test_pruned_gate_timeline_is_monotonic(self):
        from oqd_compiler_infrastructure import Pre

        from oqd_trical.backend.dynamiqs.interface import (
            DynamiqsExperiment,
            DynamiqsGate,
        )

        hilbert_space = HilbertSpace(hilbert_space=dict(E0={0, 1}))
        experiment = DynamiqsExperiment(
            frame=None,
            sequence=[
                DynamiqsGate(hamiltonian=None, duration=1e-6),
                DynamiqsGate(hamiltonian=None, duration=1e-6),
            ],
        )
        vm = Pre(DynamiqsVM(hilbert_space=hilbert_space, timestep=TIMESTEP))
        vm(experiment)
        result = vm.children[0].result

        tspan = np.array(result["tspan"])
        assert np.all(np.diff(tspan) > 0)  # strictly increasing, no double-offset
        np.testing.assert_allclose(tspan[-1], 2e-6, atol=1e-12)
        assert len(result["states"]) == len(tspan)
