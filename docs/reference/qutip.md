::: oqd_trical.backend.qutip

## Datastore output

`QutipBackend.run(...)` returns an `oqd_dataschema.Datastore` with one
`"emulation"` group. The group stores `tspan`, `states`, and `final_state` as
HDF5-serializable datasets, plus run metadata in `attrs`.

```python
backend = QutipBackend()
experiment, hilbert_space = backend.compile(circuit, fock_cutoff=4)
datastore = backend.run(experiment, hilbert_space, timestep=1e-7)

emulation = datastore["emulation"]
tspan = emulation.tspan.data
states = emulation.states.data

datastore.model_dump_hdf5("trical_run.h5")
```
