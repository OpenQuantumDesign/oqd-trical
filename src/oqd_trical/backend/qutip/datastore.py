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
"""Schema-validated result containers for oqd-trical emulator backends.

Defines :class:`TrICalEmulatorDataGroup`, a
:class:`~oqd_dataschema.GroupBase` subclass that holds the output of an
oqd-trical emulator run in the standard
:mod:`oqd_dataschema` layout (HDF5-serializable :class:`~oqd_dataschema.Datastore`).

The same group is used by the QuTiP and Dynamiqs backends so that both can be
swapped at the data layer.
"""

import json
from typing import Any, Dict, Optional

import numpy as np
from oqd_dataschema import Dataset, Datastore, GroupBase
from pydantic import ConfigDict

from oqd_trical.light_matter.compiler.analysis import HilbertSpace

########################################################################################

__all__ = [
    "TrICalEmulatorDataGroup",
    "build_emulator_datastore",
    "frame_to_array",
    "hilbert_space_to_size_dict",
    "states_to_array",
]


########################################################################################


def hilbert_space_to_size_dict(hilbert_space: HilbertSpace) -> Dict[str, int]:
    """Return the Hilbert-space subsystem sizes as a plain ``Dict[str, int]``.

    Args:
        hilbert_space: The :class:`HilbertSpace` describing the system.

    Returns:
        Mapping from subsystem label (e.g. ``"E0"``, ``"P0"``) to its
        Hilbert-space dimension.
    """
    return dict(hilbert_space.size)


def states_to_array(states) -> np.ndarray:
    """Stack a sequence of QuTiP ``Qobj`` states into a single complex array.

    The result is shape ``(n_tsteps, hilbert_dim, 1)`` when all states are kets
    (because ``Qobj.full()`` returns a column vector) and
    ``(n_tsteps, hilbert_dim, hilbert_dim)`` when all states are density
    matrices. An empty input returns a 1-D length-zero complex array, which
    is the same shape that the ``Dataset`` validator accepts.
    """
    if len(states) == 0:
        return np.empty((0,), dtype=np.complex128)

    arrays = [np.asarray(s.full()) for s in states]
    return np.stack(arrays, axis=0)


def frame_to_array(frame) -> Optional[np.ndarray]:
    """Convert a QuTiP ``QobjEvo`` (or ``Qobj``) frame to a numpy array.

    Returns ``None`` when ``frame`` is ``None`` or when the frame cannot be
    evaluated as a fixed matrix (e.g. a time-dependent ``QobjEvo`` that
    cannot be inspected at a single time); the presence of a frame is
    still recorded in the group's ``attrs``.
    """
    if frame is None:
        return None

    # Time-dependent QobjEvo: evaluate at t=0 to capture a representative
    # matrix. Static Qobj instances fall through to .full() below.
    import qutip as _qt

    if isinstance(frame, _qt.QobjEvo):
        try:
            return np.asarray(frame(0.0).full())
        except Exception:
            return None

    # Static Qobj (or anything with a .full() method).
    try:
        return np.asarray(frame.full())
    except Exception:
        return None


########################################################################################


class TrICalEmulatorDataGroup(GroupBase):
    """Standard oqd-dataschema group for the output of an oqd-trical emulator run.

    The group is intentionally minimal and shared between the QuTiP and
    Dynamiqs backends. Per :class:`~oqd_dataschema.GroupBase` semantics, all
    fields must be :class:`~oqd_dataschema.Dataset`,
    :class:`~oqd_dataschema.Table` or :class:`~oqd_dataschema.Folder`; the
    serializable metadata that does not fit those types (Hilbert-space
    layout, solver, version, …) is stored in :attr:`attrs`.

    Attributes:
        tspan: 1-D float array of length ``n_tsteps`` with the time points
            at which a state was tracked.
        states: Complex array of shape ``(n_tsteps, hilbert_dim, 1)`` for
            state-vector solvers (e.g. ``SESolver``) or
            ``(n_tsteps, hilbert_dim, hilbert_dim)`` for density-matrix
            solvers (e.g. ``MESolver``). Note that QuTiP's ``Qobj.full()``
            returns a column vector ``(N, 1)`` for kets, so the extra trailing
            dimension is always present for ket-based simulations.
        final_state: Complex array holding the state at the end of the
            evolution; 1-D ``(hilbert_dim,)`` if the solver returned kets or
            2-D ``(hilbert_dim, hilbert_dim)`` for density matrices.
            (For ket solvers, ``Qobj.full()`` returns ``(N, 1)``.)
        frame: Complex array of the rotating frame evaluated at ``t=0``,
            or ``None`` if no frame was set. The presence of a frame is
            also recorded in :attr:`attrs`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tspan: Dataset
    states: Dataset
    final_state: Dataset
    frame: Optional[Dataset] = None

    def set_attrs(
        self,
        *,
        solver: str,
        timestep: float,
        fock_cutoff,
        hilbert_space: HilbertSpace,
        frame: Any = None,
        backend: str = "qutip",
        version: str = "0.1.0",
    ) -> None:
        """Populate :attr:`attrs` with the run metadata.

        Args:
            solver: Solver name used for the run (``"SESolver"`` /
                ``"MESolver"``).
            timestep: Timestep between tracked states.
            fock_cutoff: Either an ``int`` (uniform Fock cutoff) or a
                ``Dict[str, int]`` (per-subsystem Fock cutoffs).
            hilbert_space: :class:`HilbertSpace` of the run.
            frame: Raw frame object (e.g. QuTiP ``QobjEvo``) — used only
                to decide the ``frame_present`` boolean recorded in
                :attr:`attrs`; the array is stored in the ``frame``
                :class:`Dataset` field.
            backend: Backend identifier (e.g. ``"qutip"``, ``"dynamiqs"``).
            version: oqd-trical version string.
        """
        size = hilbert_space_to_size_dict(hilbert_space)

        self.attrs["solver"] = solver
        self.attrs["timestep"] = float(timestep)
        self.attrs["fock_cutoff"] = json.dumps(
            fock_cutoff if isinstance(fock_cutoff, dict) else int(fock_cutoff)
        )
        self.attrs["hilbert_space"] = json.dumps(size)
        self.attrs["hilbert_space_labels"] = json.dumps(list(size.keys()))
        self.attrs["frame_present"] = bool(frame is not None)
        self.attrs["backend"] = backend
        self.attrs["oqd_trical_version"] = version


########################################################################################


def build_emulator_datastore(
    *,
    states,
    tspan,
    final_state,
    frame,
    hilbert_space: HilbertSpace,
    solver: str,
    timestep: float,
    fock_cutoff,
    backend: str = "qutip",
    version: str = "0.1.0",
) -> Datastore:
    """Build a :class:`Datastore` containing a single
    :class:`TrICalEmulatorDataGroup` from raw backend outputs.

    Args:
        states: Sequence of QuTiP ``Qobj`` states tracked during the run.
        tspan: Sequence of time points corresponding to ``states``.
        final_state: The QuTiP ``Qobj`` state at the end of the run.
        frame: The QuTiP ``QobjEvo`` (or ``Qobj``) frame, or ``None``.
        hilbert_space: :class:`HilbertSpace` of the run.
        solver: Solver name used for the run.
        timestep: Timestep between tracked states.
        fock_cutoff: Either an ``int`` or a ``Dict[str, int]``.
        backend: Backend identifier.
        version: oqd-trical version string.

    Returns:
        A :class:`Datastore` whose only group is a
        :class:`TrICalEmulatorDataGroup` populated with the run's data
        and metadata.
    """
    tspan_arr = np.asarray(list(tspan), dtype=np.float64)
    states_arr = states_to_array(states)
    final_arr = np.asarray(final_state.full())

    frame_arr = frame_to_array(frame)

    group = TrICalEmulatorDataGroup(
        tspan=Dataset(data=tspan_arr),
        states=Dataset(data=states_arr),
        final_state=Dataset(data=final_arr),
        frame=Dataset(data=frame_arr) if frame_arr is not None else None,
    )
    group.set_attrs(
        solver=solver,
        timestep=timestep,
        fock_cutoff=fock_cutoff,
        hilbert_space=hilbert_space,
        frame=frame,
        backend=backend,
        version=version,
    )

    return Datastore(groups={"emulation": group})
