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

from oqd_compiler_infrastructure import RewriteRule

########################################################################################


class AnalyseHilbertSpace(RewriteRule):
    """Analyses the atomic system and extracts the Hilbert space."""

    def __init__(self):
        super().__init__()

        self.hilbert_space = {}

    def map_System(self, model):
        for n, ion in enumerate(model.ions):
            self.hilbert_space[f"E{n}"] = len(ion.levels)

        for m, mode in enumerate(model.modes):
            self.hilbert_space[f"P{m}"] = None
