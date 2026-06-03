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
import json
import pathlib

import dynamiqs as dq
import numpy as np
import pytest
import qutip as qt
from oqd_core.interface.atomic import (
    AtomicCircuit,
    Beam,
    Pulse,
    SequentialProtocol,
    System,
    Yb171IIBuilder,
)
from oqd_dataschema import Datastore

from oqd_trical.backend import QutipBackend
from oqd_trical.backend.dynamiqs.vm import DynamiqsVM
from oqd_trical.backend.qutip.datastore import TrICalEmulatorDataGroup
from oqd_trical.backend.qutip.vm import QutipVM
from oqd_trical.light_matter.compiler.analysis import HilbertSpace

########################################################################################


def _microwave_circuit(duration=1e-6, rabi=2 * np.pi * 1e5):
    """Build a tiny Rabi-flop circuit on a single 171Yb+ ion."""
    Yb171 = Yb171IIBuilder().build(["q0", "q1"])
    system = System(ions=[Yb171], modes=[])
    beam = Beam(
        transition="q0->q1",
        rabi=rabi,
        detuning=0,
        phase=0,
        polarization=[0, 1, 0],
        wavevector=[1, 0, 0],
        target=0,
    )
    protocol = SequentialProtocol(sequence=[Pulse(beam=beam, duration=duration)])
    return AtomicCircuit(system=system, protocol=protocol)


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


@pytest.mark.slow
class TestQutipBackendDatastore:
    """End-to-end coverage of the new oqd-dataschema API for ``QutipBackend``.
    Marked ``@pytest.mark.slow`` because these tests require actual quantum
    dynamics integration (QuTiP 5.x stiff ODE solver), which can be slow on
    constrained CI runners. Run with ``pytest -m slow`` to include them."""

    @pytest.fixture
    def backend(self):
        # Use integrator settings that are robust enough to actually
        # integrate the time-dependent Hamiltonian emitted by
        # ``QutipCodeGeneration`` on QuTiP 5.x.
        return QutipBackend(
            solver_options={
                "progress_bar": False,
                "nsteps": 1_000_000,
                "max_step": 1e-9,
            }
        )

    def test_run_returns_datastore_with_emulation_group(self, backend):
        circuit = _microwave_circuit()
        exp, hs = backend.compile(circuit, fock_cutoff=2)
        ds = backend.run(exp, hilbert_space=hs, timestep=1e-7)

        assert isinstance(ds, Datastore)
        assert list(ds.groups.keys()) == ["emulation"]
        assert isinstance(ds["emulation"], TrICalEmulatorDataGroup)

    def test_run_datasets_have_expected_shapes(self, backend):
        circuit = _microwave_circuit()
        exp, hs = backend.compile(circuit, fock_cutoff=2)
        ds = backend.run(exp, hilbert_space=hs, timestep=1e-7)

        g = ds["emulation"]
        # tspan is 1-D, states is (n_tsteps, dim, 1) for a ket solver,
        # final_state has the same dim.
        assert g.tspan.data.ndim == 1
        assert g.states.data.ndim == 3
        assert g.states.data.shape[1:] == g.final_state.data.shape
        assert g.states.data.shape[0] == g.tspan.data.shape[0]

    def test_run_attrs_contain_run_metadata(self, backend):
        circuit = _microwave_circuit()
        exp, hs = backend.compile(circuit, fock_cutoff=2)
        ds = backend.run(exp, hilbert_space=hs, timestep=1e-7)

        attrs = ds["emulation"].attrs
        assert attrs["solver"] == "SESolver"
        assert attrs["timestep"] == pytest.approx(1e-7)
        assert attrs["backend"] == "qutip"
        assert json.loads(attrs["fock_cutoff"]) == 2
        assert "E0" in json.loads(attrs["hilbert_space"])

    def test_run_hdf5_round_trip(self, backend, tmp_path: pathlib.Path):
        circuit = _microwave_circuit()
        exp, hs = backend.compile(circuit, fock_cutoff=2)
        ds = backend.run(exp, hilbert_space=hs, timestep=1e-7)

        f = tmp_path / "qutip_emulation.h5"
        ds.model_dump_hdf5(f)

        reloaded = type(ds).model_validate_hdf5(f)
        g = reloaded["emulation"]
        assert np.allclose(g.tspan.data, ds["emulation"].tspan.data)
        assert np.allclose(g.states.data, ds["emulation"].states.data)
        assert np.allclose(g.final_state.data, ds["emulation"].final_state.data)
        assert g.attrs["solver"] == "SESolver"
        assert g.attrs["timestep"] == pytest.approx(1e-7)

    def test_run_accepts_explicit_fock_cutoff_override(self, backend):
        circuit = _microwave_circuit()
        exp, hs = backend.compile(circuit, fock_cutoff=2)
        ds = backend.run(exp, hilbert_space=hs, timestep=1e-7, fock_cutoff=5)

        assert json.loads(ds["emulation"].attrs["fock_cutoff"]) == 5
