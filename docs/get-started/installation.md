# Installation

Clone the repository using the following command:

```sh
git clone https://github.com/OpenQuantumDesign/TrICal.git
```

Install with pip:

```sh
pip install .
```

TrICal has a dependency on [midstack](https://github.com/OpenQuantumDesign/midstack), when TrICal in cloned, [midstack](https://github.com/OpenQuantumDesign/midstack) is referenced as a git submodule.

This submodule can be initialized with:

```sh
git submodule update --init --recursive
```

[midstack](https://github.com/OpenQuantumDesign/midstack) can be installed with:

```sh
cd .submodules/midstack
pip install .
```
