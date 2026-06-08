# Backends

Backends are used to execute the AtomicCircuit.

## Supported Backends

- [QuTiP](https://qutip.readthedocs.io/en/latest/) <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_trical.backend.qutip.QutipBackend] </div>
- [Dynamiqs](https://www.dynamiqs.org/) <div style="float:right;"> [![](https://img.shields.io/badge/Implementation-7C4DFF)][oqd_trical.backend.dynamiqs.DynamiqsBackend] </div>

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

## Units and solver options (Dynamiqs)

The Dynamiqs backend integrates the Schrödinger equation
$\frac{d}{dt}\lvert\psi\rangle = -i H \lvert\psi\rangle$ with $\hbar = 1$. Units flow
consistently from the atomic interface to [`dq.sesolve`](https://www.dynamiqs.org/):

- Level energies, beam Rabi frequencies and detunings are **angular frequencies
  (rad/s)**, so the lowered Hamiltonian is in rad/s.
- Pulse durations and the save `timestep` are in **seconds**.
- The phase accumulated by a time-dependent coefficient is `frequency * t`
  (rad/s × s), which is dimensionless.

Because the bare optical carrier ($\sim 2\pi\cdot 10^{15}$ rad/s) makes the equation
stiff, drive it through the
[`RotatingReferenceFrame`][oqd_trical.light_matter.compiler.approximate.RotatingReferenceFrame]
and [`RotatingWaveApprox`][oqd_trical.light_matter.compiler.approximate.RotatingWaveApprox]
passes, which remove the carrier and leave a slowly varying Hamiltonian. Accuracy
is then controlled by the explicit Diffrax tolerances rather than by rescaling time.

The Diffrax solver, its tolerances and the step-size controller are set explicitly via
[`DynamiqsSolverOptions`][oqd_trical.backend.dynamiqs.solver_options.DynamiqsSolverOptions]
(default: `Tsit5` with `rtol = atol = 1e-8`), passed as `solver_options` to the
backend. The progress meter is disabled by default to avoid the `ZMQError` raised by
`tqdm` inside Jupyter (issue #26).

<!-- prettier-ignore -->
/// admonition | Example
    type: example

```python
from oqd_trical.backend import DynamiqsBackend, DynamiqsSolverOptions

backend = DynamiqsBackend(
    solver_options=DynamiqsSolverOptions(rtol=1e-8, atol=1e-8),
)
```

///
