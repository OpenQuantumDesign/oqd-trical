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
import numpy as np
import pytest
import qutip as qt
from oqd_dataschema import Datastore, TrICalEmulatorDataGroup

from oqd_trical.backend.dynamiqs.vm import DynamiqsVM
from oqd_trical.backend.qutip.base import QutipBackend
from oqd_trical.backend.qutip.interface import QutipExperiment, QutipGate
from oqd_trical.backend.qutip.vm import QutipVM
from oqd_trical.light_matter.compiler.analysis import HilbertSpace

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


class TestQutipDatastore:
    def test_run_returns_hdf5_roundtrippable_datastore(self, tmp_path):
        hilbert_space = HilbertSpace(hilbert_space=dict(E0={0, 1}))
        experiment = QutipExperiment(
            frame=None,
            sequence=[QutipGate(hamiltonian=None, duration=1.0)],
        )

        datastore = QutipBackend(solver_options={"progress_bar": False}).run(
            experiment, hilbert_space, timestep=0.5
        )

        assert isinstance(datastore, Datastore)
        emulation = datastore["emulation"]
        assert isinstance(emulation, TrICalEmulatorDataGroup)
        np.testing.assert_allclose(emulation.tspan.data, np.array([0.0, 0.5, 1.0]))
        assert emulation.states.data.shape == (3, 2)
        assert emulation.final_state.data.shape == (2,)
        assert emulation.attrs["backend"] == "qutip"
        assert emulation.attrs["solver"] == "SESolver"
        assert emulation.attrs["hilbert_space"] == '{"E0": 2}'

        f = tmp_path / "qutip_run.h5"
        datastore.model_dump_hdf5(f)
        loaded = Datastore.model_validate_hdf5(f)

        np.testing.assert_allclose(
            loaded["emulation"].states.data, emulation.states.data
        )
