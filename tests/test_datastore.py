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

########################################################################################
import json
import pathlib

import numpy as np
import pytest
import qutip as qt
from oqd_dataschema import Dataset

from oqd_trical.backend.qutip.datastore import (
    TrICalEmulatorDataGroup,
    build_emulator_datastore,
    frame_to_array,
    hilbert_space_to_size_dict,
    states_to_array,
)
from oqd_trical.light_matter.compiler.analysis import HilbertSpace

########################################################################################


def _trivial_hilbert_space():
    return HilbertSpace(hilbert_space=dict(E0={0, 1}, P0={0, 1, 2}))


class TestHilbertSpaceToSizeDict:
    def test_returns_size_mapping(self):
        hs = _trivial_hilbert_space()
        size = hilbert_space_to_size_dict(hs)
        assert size == {"E0": 2, "P0": 3}

    def test_returns_plain_dict(self):
        hs = _trivial_hilbert_space()
        size = hilbert_space_to_size_dict(hs)
        assert isinstance(size, dict)
        for k, v in size.items():
            assert isinstance(k, str)
            assert isinstance(v, int)


class TestStatesToArray:
    def test_stacks_kets(self):
        states = [
            qt.basis(2, 0),
            qt.basis(2, 1),
            (qt.basis(2, 0) + qt.basis(2, 1)).unit(),
        ]
        arr = states_to_array(states)
        assert arr.shape == (3, 2, 1)  # qutip ket full() is (N, 1)
        assert arr.dtype == np.complex128

    def test_empty_input_returns_empty_complex_array(self):
        arr = states_to_array([])
        assert arr.shape == (0,)
        assert arr.dtype == np.complex128

    def test_density_matrices_have_extra_dim(self):
        rho0 = qt.ket2dm(qt.basis(2, 0))
        rho1 = qt.ket2dm(qt.basis(2, 1))
        arr = states_to_array([rho0, rho1])
        assert arr.shape == (2, 2, 2)


class TestFrameToArray:
    def test_none_frame_returns_none(self):
        assert frame_to_array(None) is None

    def test_qobj_returns_array(self):
        arr = frame_to_array(qt.sigmax())
        assert arr is not None
        assert arr.shape == (2, 2)


class TestTrICalEmulatorDataGroup:
    def test_group_auto_registers(self):
        from oqd_dataschema import GroupRegistry

        assert "TrICalEmulatorDataGroup" in GroupRegistry.groups

    def test_set_attrs_serializes_complex_values(self):
        g = TrICalEmulatorDataGroup(
            tspan=Dataset(data=np.array([0.0])),
            states=Dataset(data=np.zeros((1, 2, 1), dtype=np.complex128)),
            final_state=Dataset(data=np.zeros((2, 1), dtype=np.complex128)),
        )
        g.set_attrs(
            solver="SESolver",
            timestep=1e-6,
            fock_cutoff=3,
            hilbert_space=_trivial_hilbert_space(),
        )

        assert g.attrs["solver"] == "SESolver"
        assert g.attrs["timestep"] == pytest.approx(1e-6)
        # `fock_cutoff` and `hilbert_space` are JSON-serialized for HDF5 round-trip.
        assert json.loads(g.attrs["fock_cutoff"]) == 3
        assert json.loads(g.attrs["hilbert_space"]) == {"E0": 2, "P0": 3}
        assert json.loads(g.attrs["hilbert_space_labels"]) == ["E0", "P0"]
        assert g.attrs["frame_present"] is False
        assert g.attrs["backend"] == "qutip"


class TestBuildEmulatorDatastore:
    def _build(self, **overrides):
        hs = _trivial_hilbert_space()
        # Run a trivial no-op evolution: state stays |00⟩ for all times.
        H = 0 * qt.tensor(qt.sigmax(), qt.qeye(2))
        tspan = [0.0, 1.0, 2.0]
        states = [
            (-1j * H * t).expm() * qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
            for t in tspan
        ]
        kwargs = dict(
            states=states,
            tspan=tspan,
            final_state=states[-1],
            frame=None,
            hilbert_space=hs,
            solver="SESolver",
            timestep=1.0,
            fock_cutoff=3,
        )
        kwargs.update(overrides)
        return build_emulator_datastore(**kwargs)

    def test_returns_datastore_with_one_group(self):
        ds = self._build()
        assert list(ds.groups.keys()) == ["emulation"]
        assert isinstance(ds["emulation"], TrICalEmulatorDataGroup)

    def test_datasets_have_expected_shapes(self):
        ds = self._build()
        g = ds["emulation"]
        assert g.tspan.data.shape == (3,)
        assert g.states.data.shape == (3, 4, 1)  # 4 = 2*2 ion dims, 1 = ket col
        assert g.final_state.data.shape == (4, 1)

    def test_attrs_contain_run_metadata(self):
        ds = self._build(version="9.9.9")
        attrs = ds["emulation"].attrs
        assert attrs["solver"] == "SESolver"
        assert attrs["timestep"] == pytest.approx(1.0)
        assert attrs["backend"] == "qutip"
        assert attrs["oqd_trical_version"] == "9.9.9"
        assert json.loads(attrs["hilbert_space"]) == {"E0": 2, "P0": 3}

    def test_hdf5_round_trip_preserves_data_and_metadata(self, tmp_path: pathlib.Path):
        ds = self._build()
        f = tmp_path / "emulation.h5"
        ds.model_dump_hdf5(f)

        reloaded = type(ds).model_validate_hdf5(f)
        g = reloaded["emulation"]
        assert np.allclose(g.tspan.data, ds["emulation"].tspan.data)
        assert np.allclose(g.states.data, ds["emulation"].states.data)
        assert np.allclose(g.final_state.data, ds["emulation"].final_state.data)
        # Attrs round-trip.
        for k, v in ds["emulation"].attrs.items():
            assert g.attrs[k] == v
