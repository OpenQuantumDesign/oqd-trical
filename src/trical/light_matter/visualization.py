from oqd_compiler_infrastructure import ConversionRule, Post, PrettyPrint

from oqd_core.compiler.math.rules import PrintMathExpr
from oqd_core.interface.math import MathNum

from .interface.operator import OperatorAdd, CoefficientAdd

########################################################################################


class CoeffiecientPrinter(ConversionRule):
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
        return f"PrunedOperator"

    def map_Wave(self, model, operands):
        return f"exp(1j * {operands['lamb_dicke']} * (A_{model.subsystem} + C_{model.subsystem}))"

    def map_OperatorAdd(self, model, operands):
        return f"{operands['op1']} + {operands['op2']}"

    def map_OperatorMul(self, model, operands):
        return f"{'(' + operands['op1'] + ')' if isinstance(model.op1, OperatorAdd) else operands['op1']} * {'(' + operands['op2'] + ')' if isinstance(model.op2, OperatorAdd) else operands['op2']}"

    def map_OperatorKron(self, model, operands):
        return f"{'(' + operands['op1'] + ')' if isinstance(model.op1, OperatorAdd) else operands['op1']} @ {'(' + operands['op2'] + ')' if isinstance(model.op2, OperatorAdd) else operands['op2']}"

    def map_OperatorScalarMul(self, model, operands):
        return f"{'(' + operands['coeff'] + ')' if isinstance(model.coeff, CoefficientAdd) else operands['coeff']} * {'(' + operands['op'] + ')' if isinstance(model.op, OperatorAdd) else operands['op']}"

    def map_Coefficient(self, model, operands):
        return Post(CoeffiecientPrinter())(model)


class CircuitPrinter(PrettyPrint):
    """Prints An AtomicEmulatorCircuit in a pretty manner"""

    def map_Operator(self, model, operands):
        return f"Operator({Post(OperatorPrinter())(model)})"
