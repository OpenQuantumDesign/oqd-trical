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

from oqd_compiler_infrastructure import ConversionRule
from oqd_core.interface.math import MathNum

from oqd_trical.light_matter.compiler.analysis import HilbertSpace
from oqd_trical.light_matter.interface.operator import CoefficientAdd

########################################################################################


class QuantumOpticsCodeGeneration(ConversionRule):
    def __init__(self, hilbert_space: HilbertSpace):
        super().__init__()

        self.hilbert_space = hilbert_space

        _basis = ", ".join(
            [
                f"basisstate(NLevelBasis({v}), 1)"
                if k[0] == "E"
                else f"basisstate(FockBasis({v - 1}), 1)"
                for k, v in hilbert_space.size.items()
            ]
        )
        self.initial_state = f"tensor({_basis})"

    def map_AtomicEmulatorCircuit(self, model, operands):
        gates = "\n\n".join(operands["sequence"])
        return f"""
using QuantumOptics
using NPZ

psi_0 = {self.initial_state}

times = [0.0]
states = [psi_0]

{gates}

states = map(s->s.data, states)
states = transpose(hcat(states...))

npzwrite("__times.npz", times)
npzwrite("__states.npz", states)
""".strip()

    def map_AtomicEmulatorGate(self, model, operands):
        return f"""
H = (t, psi) -> {operands["hamiltonian"]}
tspan = LinRange(0, {model.duration}, 101) .+ times[end]

tout, psi_t = timeevolution.schroedinger_dynamic(tspan, states[end], H)

append!(times, tout[2:end])
append!(states, psi_t[2:end])
""".strip()

    def map_Identity(self, model, operands):
        if model.subsystem[0] == "E":
            return f"identityoperator(NLevelBasis({self.hilbert_space.size[model.subsystem]}))"

        return f"identityoperator(FockBasis({self.hilbert_space.size[model.subsystem] - 1}))"

    def map_KetBra(self, model, operands):
        return f"transition(NLevelBasis({self.hilbert_space.size[model.subsystem]}), {model.ket + 1}, {model.bra + 1})"

    def map_Annihilation(self, model, operands):
        return f"destroy(FockBasis({self.hilbert_space.size[model.subsystem] - 1}))"

    def map_Creation(self, model, operands):
        return f"create(FockBasis({self.hilbert_space.size[model.subsystem] - 1}))"

    def map_Displacement(self, model, operands):
        return f"displace(FockBasis({self.hilbert_space.size[model.subsystem] - 1}), {operands['alpha']})"

    def map_OperatorMul(self, model, operands):
        return f"{operands['op1']} * {operands['op2']}"

    def map_OperatorKron(self, model, operands):
        if isinstance(operands["op1"], list):
            return operands["op1"] + [operands["op2"]]

        return [operands["op1"], operands["op2"]]

    def map_OperatorAdd(self, model, operands):
        return f"{operands['op1']} + {operands['op2']}"

    def map_OperatorScalarMul(self, model, operands):
        if isinstance(operands["op"], list):
            op = f"tensor({', '.join(operands['op'])})"
        else:
            op = operands["op"]

        if isinstance(model.coeff, CoefficientAdd):
            return f"({operands['coeff']}) * {op}"

        return f"{operands['coeff']} * {op}"

    def map_WaveCoefficient(self, model, operands):
        if model.frequency == MathNum(value=0) and model.phase == MathNum(value=0):
            return operands["amplitude"]

        if model.frequency == MathNum(value=0):
            return f"{operands['amplitude']} * exp(1im * {operands['phase']})"

        if model.phase == MathNum(value=0):
            return f"{operands['amplitude']} * exp(1im * {operands['frequency']} * t)"

        return f"{operands['amplitude']} * exp(1im * ({operands['frequency']} * t + {operands['phase']}))"

    def map_CoefficientAdd(self, model, operands):
        return f"{operands['coeff1']} + {operands['coeff2']}"

    def map_CoefficientMul(self, model, operands):
        return f"{operands['coeff1']} * {operands['coeff2']}"

    def map_MathNum(self, model, operands):
        return f"{model.value}"

    def map_MathImag(self, model, operands):
        return "1im"

    def map_MathVar(self, model, operands):
        if model.name == "t":
            return "t"

        raise ValueError(
            f"Unsupported variable {model.name}, only variable t is supported"
        )

    def map_MathFunc(self, model, operands):
        if model.func == "heaviside":
            heaviside_fn = "(x -> 0.5 * (sign(x) + 1)"
            return f"{heaviside_fn}({operands['expr2']})"

        return f"{model.func}({operands['expr2']})"

    def map_MathAdd(self, model, operands):
        return f"{operands['expr1']} + {operands['expr2']}"

    def map_MathMul(self, model, operands):
        return f"{operands['expr1']} * {operands['expr2']}"

    def map_MathPow(self, model, operands):
        return f"{operands['expr1']} ^ {operands['expr2']}"
