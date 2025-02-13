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

from oqd_compiler_infrastructure import RewriteRule, TypeReflectBaseModel
from typing import Dict, Optional, Set
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
        return HilbertSpace(hilbert_space=self._hilbert_space)

    def map_System(self, model):
        for n, ion in enumerate(model.ions):
            self._hilbert_space[f"E{n}"] = set(range(ion.levels))

        for m, mode in enumerate(model.modes):
            self._hilbert_space[f"P{m}"] = None

    def map_KetBra(self, model):
        if model.subsystem not in self._hilbert_space.keys():
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

        if model.subsystem[0] == "E":
            self._hilbert_space[model.subsystem] = set()
            return

        if model.subsystem[0] == "P":
            self._hilbert_space[model.subsystem] = None
            return
