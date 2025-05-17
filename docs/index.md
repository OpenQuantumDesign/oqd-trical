# 

<p align="center">
  <img src="img/oqd-logo-black.png#only-light" alt="Logo" style="max-height: 200px;">
  <img src="img/oqd-logo-white.png#only-dark" alt="Logo" style="max-height: 200px;">
</p>

<div align="center">
    <h2 align="center">
    Open Quantum Design: TrICal (Trapped-Ion Calculator)
    </h2>
</div>

<!-- [![PyPI Version](https://img.shields.io/pypi/v/oqd-core)](https://pypi.org/project/oqd-core)
[![CI](https://github.com/OpenQuantumDesign/oqd-core/actions/workflows/pytest.yml/badge.svg)](https://github.com/OpenQuantumDesign/oqd-core/actions/workflows/pytest.yml) -->
![versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)


<!-- prettier-ignore -->
/// admonition | Warning
    type: warning
TrICal is still currently in heavy development, breaking changes might be made!
Refer to [Features in development](get-started/outlook.md).
///

Given a description of a trapped-ion experiment with:

- $N$ ions in a chain
  - $J$ specified levels each
- $M$ lasers
- $L$ motional modes

<!-- prettier-ignore -->
/// admonition | Goal
    type: goal

1. TrICal constructs the system Hamiltonian.
2. Applies approximations to the system Hamiltonian.
3. Connects with a quantum simulation backend (e.g. QuTiP) to perform simulations.

![](figures/pipeline.png)
///

TrICal is developed under [Open Quantum Design (OQD)](https://openquantumdesign.org/) as a component of the OQD open-source full stack quantum computer with trapped-ions.

![](figures/stack_diagram.png)
