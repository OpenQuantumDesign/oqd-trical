# ![Open Quantum Design](https://raw.githubusercontent.com/OpenQuantumDesign/oqd-compiler-infrastructure/main/docs/img/oqd-logo-text.png)

<h2 align="center">
    Open Quantum Design: TrICal
</h2>

[![doc](https://img.shields.io/badge/documentation-lightblue)](https://docs.openquantumdesign.org/open-quantum-design-core)
[![PyPI Version](https://img.shields.io/pypi/v/oqd-core)](https://pypi.org/project/oqd-core)
[![CI](https://github.com/OpenQuantumDesign/oqd-core/actions/workflows/pytest.yml/badge.svg)](https://github.com/OpenQuantumDesign/oqd-core/actions/workflows/pytest.yml)![versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

## Installation <a name="installation"></a>

```bash
pip install oqd-core
```

or through `git`,

```bash
pip install git+https://github.com/OpenQuantumDesign/oqd-core.git
```

To develop, clone the repository locally:

```bash
git clone https://github.com/OpenQuantumDesign/oqd-core
pip install .
```

## Getting Started <a name="Getting Started"></a>

## Documentation <a name="documentation"></a>

Documentation is implemented with [MkDocs](https://www.mkdocs.org/).
To install the dependencies for documentation, run:

```
pip install -e ".[docs]"
```

To deploy the documentation server locally:

```
cp -r examples/ docs/examples/
mkdocs serve
```

### Where in the stack

```mermaid
block-beta
   columns 3

   block:Interface
       columns 1
       InterfaceTitle("<i><b>Interfaces</b><i/>")
       InterfaceDigital["<b>Digital Interface</b>\nQuantum circuits with discrete gates"]
       space
       InterfaceAnalog["<b>Analog Interface</b>\n Continuous-time evolution with Hamiltonians"]
       space
       InterfaceAtomic["<b>Atomic Interface</b>\nLight-matter interactions between lasers and ions"]
       space
    end

    block:IR
       columns 1
       IRTitle("<i><b>IRs</b><i/>")
       IRDigital["Quantum circuit IR\nopenQASM, LLVM+QIR"]
       space
       IRAnalog["openQSIM"]
       space
       IRAtomic["openAPL"]
       space
    end

    block:Emulator
       columns 1
       EmulatorsTitle("<i><b>Classical Emulators</b><i/>")

       EmulatorDigital["Pennylane, Qiskit"]
       space
       EmulatorAnalog["QuTiP, QuantumOptics.jl"]
       space
       EmulatorAtomic["TrICal, QuantumIon.jl"]
       space
    end

    space
    block:RealTime
       columns 1
       RealTimeTitle("<i><b>Real-Time</b><i/>")
       space
       RTSoftware["ARTIQ, DAX, OQDAX"]
       space
       RTGateware["Sinara Real-Time Control"]
       space
       RTHardware["Lasers, Modulators, Photodetection, Ion Trap"]
       space
       RTApparatus["Trapped-Ion QPU (<sup>171</sup>Yb<sup>+</sup>, <sup>133</sup>Ba<sup>+</sup>)"]
       space
    end
    space

   InterfaceDigital --> IRDigital
   InterfaceAnalog --> IRAnalog
   InterfaceAtomic --> IRAtomic

   IRDigital --> IRAnalog
   IRAnalog --> IRAtomic

   IRDigital --> EmulatorDigital
   IRAnalog --> EmulatorAnalog
   IRAtomic --> EmulatorAtomic

   IRAtomic --> RealTimeTitle

   RTSoftware --> RTGateware
   RTGateware --> RTHardware
   RTHardware --> RTApparatus

    classDef title fill:#d6d4d4,stroke:#333,color:#333;
    classDef digital fill:#E7E08B,stroke:#333,color:#333;
    classDef analog fill:#E4E9B2,stroke:#333,color:#333;
    classDef atomic fill:#D2E4C4,stroke:#333,color:#333;
    classDef realtime fill:#B5CBB7,stroke:#333,color:#333;

    classDef highlight fill:#f2bbbb,stroke:#333,color:#333,stroke-dasharray: 5 5;

    class InterfaceTitle,IRTitle,EmulatorsTitle,RealTimeTitle title
    class InterfaceDigital,IRDigital,EmulatorDigital digital
    class InterfaceAnalog,IRAnalog,EmulatorAnalog analog
    class InterfaceAtomic,IRAtomic,EmulatorAtomic atomic
    class RTSoftware,RTGateware,RTHardware,RTApparatus realtime

   class EmulatorAtomic highlight
```

The stack components highlighted in red are contained in this repository.
