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
