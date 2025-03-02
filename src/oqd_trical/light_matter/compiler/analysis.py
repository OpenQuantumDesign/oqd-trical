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

from typing import Dict, Optional, Set

from oqd_compiler_infrastructure import RewriteRule, TypeReflectBaseModel
from oqd_core.interface.atomic import Level
from pydantic import ConfigDict

########################################################################################


class HilbertSpace(TypeReflectBaseModel):
    model_config = ConfigDict(frozen=True)
    hilbert_space: Dict[str, Optional[Set[int]]]

    @property
    def size(self):
        return {
            k: len(v) if isinstance(v, set) else v
            for k, v in self.hilbert_space.items()
        }

    def get_relabel_rules(self):
        relabel_rules = {}
        for k, v in self.hilbert_space.items():
            if k[0] == "E":
                relabel_rules[k] = {i: n for n, i in enumerate(v)}
        return relabel_rules


class GetHilbertSpace(RewriteRule):
    def __init__(self):
        self._hilbert_space = {}

    @property
    def hilbert_space(self):
        hilbert_space = {}
        for k, v in self._hilbert_space.items():
            if k[0] == "E" and v is None:
                hilbert_space[k] = {0}
            else:
                hilbert_space[k] = v
        return HilbertSpace(hilbert_space=hilbert_space)

    def map_System(self, model):
        for n, ion in enumerate(model.ions):
            self._hilbert_space[f"E{n}"] = set(range(len(ion.levels)))

        for m, mode in enumerate(model.modes):
            self._hilbert_space[f"P{m}"] = None

    def map_KetBra(self, model):
        if (
            model.subsystem not in self._hilbert_space.keys()
            or self._hilbert_space[model.subsystem] is None
        ):
            self._hilbert_space[model.subsystem] = set()

        self._hilbert_space[model.subsystem].update((model.ket, model.bra))

    def map_Annihilation(self, model):
        if model.subsystem in self._hilbert_space.keys():
            return

        self._hilbert_space[model.subsystem] = None

    def map_Creation(self, model):
        if model.subsystem in self._hilbert_space.keys():
            return

        self._hilbert_space[model.subsystem] = None

    def map_Displacement(self, model):
        if model.subsystem in self._hilbert_space.keys():
            return

        self._hilbert_space[model.subsystem] = None

    def map_Identity(self, model):
        if model.subsystem in self._hilbert_space.keys():
            return

        self._hilbert_space[model.subsystem] = None


########################################################################################


class ExtractTimeScales(RewriteRule):
    def __init__(self):
        self._timescales = set()

    @property
    def timescales(self):
        return self._timescales

    def map_Level(self, model):
        self._timescales.add(model.energy)

    def map_Transition(self, model):
        if isinstance(model.level1, Level) and isinstance(model.level2, Level):
            self._timescales.add(model.level2.energy - model.level1.energy)

    def map_Phonon(self, model):
        self._timescales.add(model.energy)

    def map_Beam(self, model):
        self._timescales.add(model.rabi)
        self._timescales.add(model.detuning)

    def map_Pulse(self, model):
        self._timescales.add(1 / model.duration)

    def map_WaveCoefficient(self, model):
        self._timescales.add(model.amplitude)
        self._timescales.add(model.frequency)

    def map_AtomicEmulatorGate(self, model):
        self._timescales.add(1 / model.duration)
