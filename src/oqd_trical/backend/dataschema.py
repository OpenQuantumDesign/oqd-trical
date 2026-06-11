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

import json
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping

import numpy as np
import qutip as qt

if not hasattr(np.dtypes, "StringDType"):
    np.dtypes.StringDType = np.dtypes.StrDType

from oqd_dataschema.dataset import Dataset
from oqd_dataschema.datastore import Datastore
from oqd_dataschema.group import GroupBase

########################################################################################

EMULATION_GROUP_KEY = "emulation"

__all__ = [
    "EMULATION_GROUP_KEY",
    "TrICalEmulatorDataGroup",
    "emulation_result_to_datastore",
]


class TrICalEmulatorDataGroup(GroupBase):
    """Standard oqd-dataschema group for TrICal atomic emulator output."""

    tspan: Dataset
    states: Dataset
    final_state: Dataset


def _state_to_array(state: qt.Qobj) -> np.ndarray:
    data = np.asarray(state.full())
    if state.isoper:
        return data
    return data.reshape(-1)


def _oqd_trical_version() -> str:
    try:
        return version("oqd-trical")
    except PackageNotFoundError:
        return "unknown"


def emulation_result_to_datastore(
    result: Mapping[str, Any],
    *,
    solver: str,
    timestep: float,
    backend: str = "qutip",
) -> Datastore:
    """Build a datastore from a QutipVM or DynamiqsVM result mapping."""
    tspan = np.asarray(result["tspan"], dtype=float)
    states = np.stack([_state_to_array(state) for state in result["states"]])
    final_state = _state_to_array(result["final_state"])

    hilbert_space = result["hilbert_space"]
    frame = result.get("frame")

    group = TrICalEmulatorDataGroup(
        tspan=Dataset(data=tspan),
        states=Dataset(data=states),
        final_state=Dataset(data=final_state),
        attrs={
            "solver": solver,
            "timestep": float(timestep),
            "hilbert_space": json.dumps(hilbert_space.size),
            "backend": backend,
            "oqd_trical_version": _oqd_trical_version(),
            "frame": "" if frame is None else str(frame),
            "frame_is_none": int(frame is None),
        },
    )
    return Datastore(groups={EMULATION_GROUP_KEY: group})
