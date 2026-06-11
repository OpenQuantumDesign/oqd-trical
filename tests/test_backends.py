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
from pathlib import Path

import dynamiqs as dq
import numpy as np
import pytest
import qutip as qt

from oqd_trical.backend.dataschema import (
    EMULATION_GROUP_KEY,
    emulation_result_to_datastore,
)
from oqd_trical.backend.dynamiqs.vm import DynamiqsVM
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


class TestEmulationDatastore:
    def test_qutip_result_round_trip(self, tmp_path: Path):
        hilbert_space = HilbertSpace(hilbert_space=dict(E0={0, 1}, E1={0, 1}))
        initial_state = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
        vm = QutipVM(
            hilbert_space=hilbert_space,
            timestep=1,
            initial_state=initial_state,
            solver_options={},
        )
        vm.frame = None

        datastore = emulation_result_to_datastore(
            vm.result,
            solver="SESolver",
            timestep=1.0,
            backend="qutip",
        )

        sim = datastore.groups[EMULATION_GROUP_KEY]
        assert sim.tspan.data.shape == (1,)
        assert sim.states.data.shape == (1, 4)
        assert sim.final_state.data.shape == (4,)
        assert json.loads(sim.attrs["hilbert_space"]) == hilbert_space.size
        assert sim.attrs["solver"] == "SESolver"
        assert sim.attrs["timestep"] == 1.0

        filepath = tmp_path / "trical_run.h5"
        datastore.model_dump_hdf5(filepath)
        loaded = datastore.model_validate_hdf5(filepath)

        np.testing.assert_allclose(
            loaded.groups[EMULATION_GROUP_KEY].tspan.data,
            sim.tspan.data,
        )
        np.testing.assert_allclose(
            loaded.groups[EMULATION_GROUP_KEY].states.data,
            sim.states.data,
        )
        np.testing.assert_allclose(
            loaded.groups[EMULATION_GROUP_KEY].final_state.data,
            sim.final_state.data,
        )
        assert loaded.groups[EMULATION_GROUP_KEY].attrs == sim.attrs
