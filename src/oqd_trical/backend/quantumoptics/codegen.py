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

########################################################################################
from oqd_trical.light_matter.compiler.analysis import HilbertSpace

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

npzwrite("times.npz", times)
npzwrite("states.npz", states)
""".strip()

    def map_AtomicEmulatorGate(self, model, operands):
        return f"""
H = {operands["hamiltonian"]}
tspan = LinRange(0, {model.duration}, 101) .+ times[end]

tout, psi_t = timeevolution.schroedinger_dynamic(tspan, states[end], H)

append!(times, tout[2:end])
append!(states, psi_t[2:end])
""".strip()

    def map_Identity(self, model, operands):
        if model.subsystem[0] == "E":
            return f"((t, psi) -> identityoperator(NLevelBasis({self.hilbert_space.size[model.subsystem]})))"

        return f"((t, psi) -> identityoperator(FockBasis({self.hilbert_space.size[model.subsystem] - 1})))"

    def map_KetBra(self, model, operands):
        return f"((t, psi) -> transition(NLevelBasis({self.hilbert_space.size[model.subsystem]}), {model.ket + 1}, {model.bra + 1}))"

    def map_Annihilation(self, model, operands):
        return f"((t, psi) -> destroy(FockBasis({self.hilbert_space.size[model.subsystem] - 1})))"

    def map_Creation(self, model, operands):
        return f"((t, psi) -> create(FockBasis({self.hilbert_space.size[model.subsystem] - 1})))"

    def map_Displacement(self, model, operands):
        return f"((t, psi) -> displace(FockBasis({self.hilbert_space.size[model.subsystem] - 1}), {operands['alpha']}(t, psi)))"

    def map_OperatorMul(self, model, operands):
        return f"((t, psi) -> {operands['op1']}(t, psi) * {operands['op2']}(t, psi))"

    def map_OperatorKron(self, model, operands):
        return f"((t, psi) -> tensor({operands['op1']}(t, psi), {operands['op2']}(t, psi)))"

    def map_OperatorAdd(self, model, operands):
        return f"((t, psi) -> {operands['op1']}(t, psi) + {operands['op2']}(t, psi))"

    def map_OperatorScalarMul(self, model, operands):
        return f"((t, psi) -> {operands['coeff']}(t, psi) * {operands['op']}(t, psi))"

    def map_WaveCoefficient(self, model, operands):
        return f"((t, psi) -> {operands['amplitude']}(t, psi) * exp(1im * {operands['frequency']}(t, psi) * t + {operands['phase']}(t, psi)))"

    def map_CoefficientAdd(self, model, operands):
        return (
            f"((t, psi) -> {operands['coeff1']}(t, psi) + {operands['coeff2']}(t, psi))"
        )

    def map_CoefficientMul(self, model, operands):
        return (
            f"((t, psi) -> {operands['coeff1']}(t, psi) * {operands['coeff2']}(t, psi))"
        )

    def map_MathNum(self, model, operands):
        return f"((t, psi) -> {model.value})"

    def map_MathImag(self, model, operands):
        return "((t, psi) -> 1im)"

    def map_MathVar(self, model, operands):
        if model.name == "t":
            return "((t, psi) -> t)"

        raise ValueError(
            f"Unsupported variable {model.name}, only variable t is supported"
        )

    def map_MathFunc(self, model, operands):
        if model.func == "heaviside":
            heaviside_fn = "(x -> 0.5 * (sign(x) + 1))"
            return f"((t, psi) -> {heaviside_fn}({operands['expr2']}(t, psi)))"

        return f"((t, psi) -> {model.func}({operands['expr2']}(t, psi)))"

    def map_MathAdd(self, model, operands):
        return (
            f"((t, psi) -> {operands['expr1']}(t, psi) + {operands['expr2']}(t, psi))"
        )

    def map_MathSub(self, model, operands):
        return (
            f"((t, psi) -> {operands['expr1']}(t, psi) - {operands['expr2']}(t, psi))"
        )

    def map_MathMul(self, model, operands):
        return (
            f"((t, psi) -> {operands['expr1']}(t, psi) * {operands['expr2']}(t, psi))"
        )

    def map_MathDiv(self, model, operands):
        return (
            f"((t, psi) -> {operands['expr1']}(t, psi) / {operands['expr2']}(t, psi))"
        )

    def map_MathPow(self, model, operands):
        return (
            f"((t, psi) -> {operands['expr1']}(t, psi) ^ {operands['expr2']}(t, psi))"
        )
