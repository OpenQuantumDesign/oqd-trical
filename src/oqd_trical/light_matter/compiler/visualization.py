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

from matplotlib import pyplot as plt
from oqd_compiler_infrastructure import ConversionRule, Post, PrettyPrint, RewriteRule
from oqd_core.compiler.math.rules import PrintMathExpr
from oqd_core.interface.math import MathNum

from oqd_trical.light_matter.interface.operator import CoefficientAdd, OperatorAdd

########################################################################################


class CoefficientPrinter(ConversionRule):
    """Prints Coefficients in a pretty manner"""

    def map_MathExpr(self, model, operands):
        return Post(PrintMathExpr())(model)

    def map_MathAdd(self, model, operands):
        return f"({Post(PrintMathExpr())(model)})"

    def map_WaveCoefficient(self, model, operands):
        frequency_term = (
            f"{operands['frequency']} * t"
            if model.frequency != MathNum(value=0)
            else ""
        )
        phase_term = f"{operands['phase']}" if model.phase != MathNum(value=0) else ""

        wave_term = (
            f" * exp(1j * ({frequency_term}{' + ' if frequency_term and phase_term else ''}{phase_term}))"
            if phase_term or frequency_term
            else ""
        )

        return f"{operands['amplitude']}{wave_term}"

    def map_CoefficientAdd(self, model, operands):
        return f"{'(' + operands['coeff1'] + ')' if isinstance(model.coeff1, CoefficientAdd) else operands['coeff1']} + {'(' + operands['coeff2'] + ')' if isinstance(model.coeff2, CoefficientAdd) else operands['coeff2']}"

    def map_CoefficientMul(self, model, operands):
        return f"{'(' + operands['coeff1'] + ')' if isinstance(model.coeff1, CoefficientAdd) else operands['coeff1']} * {'(' + operands['coeff2'] + ')' if isinstance(model.coeff2, CoefficientAdd) else operands['coeff2']}"


class OperatorPrinter(ConversionRule):
    """Prints Operators in a pretty manner"""

    def map_KetBra(self, model, operands):
        return f"|{model.ket}><{model.bra}|_{model.subsystem}"

    def map_Annihilation(self, model, operands):
        return f"A_{model.subsystem}"

    def map_Creation(self, model, operands):
        return f"C_{model.subsystem}"

    def map_Identity(self, model, operands):
        return f"I_{model.subsystem}"

    def map_PrunedOperator(self, model, operands):
        return "PrunedOperator"

    def map_Displacement(self, model, operands):
        return f"D({operands['alpha']})_{model.subsystem}"

    def map_OperatorAdd(self, model, operands):
        return f"{operands['op1']} + {operands['op2']}"

    def map_OperatorMul(self, model, operands):
        return f"{'(' + operands['op1'] + ')' if isinstance(model.op1, OperatorAdd) else operands['op1']} * {'(' + operands['op2'] + ')' if isinstance(model.op2, OperatorAdd) else operands['op2']}"

    def map_OperatorKron(self, model, operands):
        return f"{'(' + operands['op1'] + ')' if isinstance(model.op1, OperatorAdd) else operands['op1']} @ {'(' + operands['op2'] + ')' if isinstance(model.op2, OperatorAdd) else operands['op2']}"

    def map_OperatorScalarMul(self, model, operands):
        return f"{'(' + operands['coeff'] + ')' if isinstance(model.coeff, CoefficientAdd) else operands['coeff']} * {'(' + operands['op'] + ')' if isinstance(model.op, OperatorAdd) else operands['op']}"

    def map_Coefficient(self, model, operands):
        return Post(CoefficientPrinter())(model)


class CondensedOperatorPrettyPrint(PrettyPrint):
    """Prints An AtomicEmulatorCircuit in a pretty manner"""

    def map_Operator(self, model, operands):
        return f"Operator({Post(OperatorPrinter())(model)})"


########################################################################################'


class PlotAtomicCircuit(RewriteRule):
    def __init__(self, t=0, *, ax=None):
        super().__init__()

        if ax:
            self.ax = ax
        else:
            self.ax = plt.subplots()

    @property
    def fig(self):
        return self.ax.get_figure()

    def map_Level(self, model):
        self.ax.plot(
            [model.orbital, model.orbital + 1],
            [model.energy, model.energy],
            color="k",
            alpha=0.5,
        )

    def map_Transition(self, model):
        self.ax.plot(
            [model.level1.orbital + 0.5, model.level2.orbital + 0.5],
            [model.level1.energy, model.level2.energy],
            color="k",
            ls="--",
            alpha=0.5,
        )

    def map_Beam(self, model):
        transition = model.transition

        if isinstance(model.detuning, MathNum):
            self.ax.plot(
                [transition.level1.orbital + 0.5, transition.level2.orbital + 0.5],
                [
                    transition.level1.energy,
                    transition.level2.energy + model.detuning.value,
                ],
                color="k",
                ls="-",
                alpha=0.5,
            )
