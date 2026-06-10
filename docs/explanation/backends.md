# Backends

Backends are used to execute the AtomicCircuit.

## Supported Backends

- [QuTiP](https://qutip.readthedocs.io/en/latest/) <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_trical.backend.qutip.QutipBackend] </div>
- [Dynamiqs](https://qutip.readthedocs.io/en/latest/) <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_trical.backend.dynamiqs.DynamiqsBackend] </div>

## Compile

Compiles the AtomicCircuit into a compatible form for the backend to run on.

<!-- prettier-ignore -->
/// admonition | Examples
    type: example

- QuTiP requires the AtomicCircuit be compiled to a [`QutipExperiment`][oqd_trical.backend.qutip.interface.QutipExperiment].
- Dynamiqs requires the AtomicCircuit be compiled to a [`DynamiqsExperiment`][oqd_trical.backend.dynamiqs.interface.DynamiqsExperiment].

///

## Run

Executes the compatible form of the AtomicCircuit with the backend using a tree walking interpreter.

<!-- prettier-ignore -->
/// admonition | Examples
    type: example

- QuTiP uses [`QutipVM`][oqd_trical.backend.qutip.vm.QutipVM] as its tree walking interpreter.
- Dynamiqs uses [`DynamiqsVM`][oqd_trical.backend.dynamiqs.vm.DynamiqsVM] as its tree walking interpreter.

///

## Results

The result of a backend run is a schema-validated `oqd_dataschema.Datastore`
that wraps a single
[`TrICalEmulatorDataGroup`][oqd_trical.backend.qutip.datastore.TrICalEmulatorDataGroup]
named `"emulation"`. The group contains:

- `tspan` — 1-D float array of the time points at which a state was tracked.
- `states` — complex array of shape `(n_tsteps, hilbert_dim, 1)` for state-vector
  solvers (QuTiP's ``Qobj.full()`` returns a column vector for kets) or
  `(n_tsteps, hilbert_dim, hilbert_dim)` for density-matrix solvers.
- `final_state` — complex array of the state at the end of the evolution;
  shape `(hilbert_dim, 1)` for ket solvers, `(hilbert_dim, hilbert_dim)` for
  density-matrix solvers.
- `frame` — optional complex array of the rotating frame evaluated at `t=0`.

Run metadata (solver, timestep, fock cutoff, Hilbert-space layout, frame
presence, backend identifier, oqd-trical version) is stored in the group's
`attrs` and survives an HDF5 round-trip:

```python
import pathlib
from oqd_trical.backend import QutipBackend

# ... build circuit, compile, run ...
datastore = backend.run(experiment, hilbert_space=hilbert_space, timestep=1e-8)

# Save to disk in the standard oqd-dataschema HDF5 format
datastore.model_dump_hdf5(pathlib.Path("emulation.h5"))

# Reload (note the exact same dictionary access pattern)
reloaded = type(datastore).model_validate_hdf5(pathlib.Path("emulation.h5"))
tspan = reloaded["emulation"].tspan.data
states = reloaded["emulation"].states.data
```

See `tests/test_datastore.py` and `tests/test_backends.py::TestQutipBackendDatastore`
for end-to-end usage examples.
