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

import pytest

import numpy as np

from oqd_core.interface.atomic import (
    Level,
    Transition,
    Ion,
    Phonon,
    System,
    Beam,
    Pulse,
    ParallelProtocol,
    SequentialProtocol,
    AtomicCircuit,
)
from oqd_trical.backend.qutip import QutipBackend

########################################################################################


@pytest.fixture
def system():
    downstate = Level(
        principal=6,
        spin=1 / 2,
        orbital=0,
        nuclear=1 / 2,
        spin_orbital=1 / 2,
        spin_orbital_nuclear=0,
        spin_orbital_nuclear_magnetization=0,
        energy=2 * np.pi * 0,
        label="q0",
    )
    upstate = Level(
        principal=6,
        spin=1 / 2,
        orbital=0,
        nuclear=1 / 2,
        spin_orbital=1 / 2,
        spin_orbital_nuclear=1,
        spin_orbital_nuclear_magnetization=0,
        energy=2 * np.pi * 10,
        label="q1",
    )
    estate = Level(
        principal=5,
        spin=1 / 2,
        orbital=1,
        nuclear=1 / 2,
        spin_orbital=1 / 2,
        spin_orbital_nuclear=1,
        spin_orbital_nuclear_magnetization=-1,
        energy=2 * np.pi * 100,
        label="e0",
    )
    estate2 = Level(
        principal=5,
        spin=1 / 2,
        orbital=1,
        nuclear=1 / 2,
        spin_orbital=1 / 2,
        spin_orbital_nuclear=1,
        spin_orbital_nuclear_magnetization=1,
        energy=2 * np.pi * 110,
        label="e1",
    )

    transitions = [
        Transition(
            level1=downstate, level2=estate, einsteinA=1, multipole="E1", label="q0->e0"
        ),
        Transition(
            level1=downstate,
            level2=estate2,
            einsteinA=1,
            multipole="E1",
            label="q0->e1",
        ),
        Transition(
            level1=upstate, level2=estate, einsteinA=1, multipole="E1", label="q1->e0"
        ),
        Transition(
            level1=upstate, level2=estate2, einsteinA=1, multipole="E1", label="q1->e1"
        ),
    ]

    ion = Ion(
        mass=171,
        charge=1,
        position=[0, 0, 0],
        levels=[downstate, upstate, estate, estate2],
        transitions=transitions,
    )

    COM_x = Phonon(energy=0.1, eigenvector=[1, 0, 0])

    system = System(
        ions=[
            ion,
        ],
        modes=[
            COM_x,
        ],
    )
    return system


def test_stationary(system):
    beam = Beam(
        transition=system.ions[0].transitions[0],
        rabi=0,
        detuning=0,
        phase=0,
        wavevector=[1, 0, 0],
        polarization=[0, 1, 0],
        target=0,
    )

    protocol = SequentialProtocol(sequence=[Pulse(beam=beam, duration=1)])

    circuit = AtomicCircuit(system=system, protocol=protocol)

    backend = QutipBackend()

    experiment, hilbert_space = backend.compile(circuit=circuit, fock_cutoff=3)

    res = backend.run(experiment=experiment, hilbert_space=hilbert_space, timestep=1)


def test_direct_transition(system):
    beam = Beam(
        transition=system.ions[0].transitions[0],
        rabi=1,
        detuning=0,
        phase=0,
        wavevector=[1, 0, 0],
        polarization=[0, 1, 0],
        target=0,
    )

    protocol = SequentialProtocol(sequence=[Pulse(beam=beam, duration=1)])

    circuit = AtomicCircuit(system=system, protocol=protocol)

    backend = QutipBackend()

    experiment, hilbert_space = backend.compile(circuit=circuit, fock_cutoff=3)

    res = backend.run(experiment=experiment, hilbert_space=hilbert_space, timestep=1)
