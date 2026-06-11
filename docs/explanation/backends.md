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

[`QutipBackend.run`][oqd_trical.backend.qutip.QutipBackend.run] returns an
[`oqd_dataschema.Datastore`][oqd_dataschema.datastore.Datastore] containing a
[`TrICalEmulatorDataGroup`][oqd_trical.backend.dataschema.TrICalEmulatorDataGroup]
under the `emulation` key:

```python
import json

backend = QutipBackend(approx_pass=approx_pass)
experiment, hilbert_space = backend.compile(circuit, fock_cutoff=4)
datastore = backend.run(experiment, hilbert_space, timestep=1e-7)

datastore.model_dump_hdf5("trical_run.h5")

sim = datastore.groups["emulation"]
tspan = sim.tspan.data
states = sim.states.data
hilbert_dims = json.loads(sim.attrs["hilbert_space"])
```

<!-- prettier-ignore -->
/// admonition | Examples
    type: example

- QuTiP uses [`QutipVM`][oqd_trical.backend.qutip.vm.QutipVM] as its tree walking interpreter.
- Dynamiqs uses [`DynamiqsVM`][oqd_trical.backend.dynamiqs.vm.DynamiqsVM] as its tree walking interpreter.

///
