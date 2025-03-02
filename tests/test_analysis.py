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

########################################################################################
import numpy as np
import pytest
from oqd_compiler_infrastructure import Post
from oqd_core.interface.atomic import (
    Level,
    Transition,
)

from oqd_trical.light_matter.compiler.analysis import ExtractTimeScales

########################################################################################


class TestExtractTimeScales:
    @pytest.mark.parametrize(
        "input,expected",
        [
            (
                Level(
                    principal=6,
                    spin=1 / 2,
                    orbital=0,
                    nuclear=1 / 2,
                    spin_orbital=1 / 2,
                    spin_orbital_nuclear=0,
                    spin_orbital_nuclear_magnetization=0,
                    energy=0,
                    label="q0",
                ),
                {0},
            ),
            (
                Level(
                    principal=6,
                    spin=1 / 2,
                    orbital=0,
                    nuclear=1 / 2,
                    spin_orbital=1 / 2,
                    spin_orbital_nuclear=1,
                    spin_orbital_nuclear_magnetization=0,
                    energy=2 * np.pi * 12.643e9,
                    label="q1",
                ),
                {2 * np.pi * 12.643e9},
            ),
            (
                Transition(
                    level1="q0",
                    level2="q1",
                    einsteinA=1,
                    multipole="M1",
                    label="q0->q1",
                ),
                set(),
            ),
            (
                Transition(
                    level1=Level(
                        principal=6,
                        spin=1 / 2,
                        orbital=0,
                        nuclear=1 / 2,
                        spin_orbital=1 / 2,
                        spin_orbital_nuclear=0,
                        spin_orbital_nuclear_magnetization=0,
                        energy=0,
                        label="q0",
                    ),
                    level2=Level(
                        principal=6,
                        spin=1 / 2,
                        orbital=0,
                        nuclear=1 / 2,
                        spin_orbital=1 / 2,
                        spin_orbital_nuclear=1,
                        spin_orbital_nuclear_magnetization=0,
                        energy=2 * np.pi * 12.643e9,
                        label="q1",
                    ),
                    einsteinA=1,
                    multipole="M1",
                    label="q0->q1",
                ),
                {0, 2 * np.pi * 12.643e9},
            ),
        ],
    )
    def test_timescale_extraction(self, input, expected):
        extract_time_scales_rule = ExtractTimeScales()
        analysis_pass = Post(extract_time_scales_rule)

        analysis_pass(input)

        assert extract_time_scales_rule.timescales == expected
