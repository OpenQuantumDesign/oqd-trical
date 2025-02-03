from __future__ import annotations

from typing import Union, Annotated

from pydantic import AfterValidator

from oqd_compiler_infrastructure import TypeReflectBaseModel
from oqd_core.interface.math import CastMathExpr


########################################################################################


class Coefficient(TypeReflectBaseModel):
    """
    Class representing a scalar coefficient for an operator.
    """

    def __neg__(self):
        return CoefficientMul(
            coeff1=WaveCoefficient(amplitude=-1, frequency=0, phase=0),
            coeff2=self,
        )

    def __pos__(self):
        return self

    def __add__(self, other):
        return CoefficientAdd(coeff1=self, coeff2=other)

    def __sub__(self, other):
        return OperatorAdd(
            op1=self,
            op2=CoefficientMul(
                coeff1=WaveCoefficient(amplitude=-1, frequency=0, phase=0),
                coeff2=other,
            ),
        )

    def __mul__(self, other):
        if isinstance(other, Coefficient):
            return CoefficientMul(coeff1=self, coeff2=other)
        else:
            return other * self

    def __rmul__(self, other):
        return self * other

    pass


########################################################################################


class WaveCoefficient(Coefficient):
    """
    Class representing a wave coefficient for an operator of the following form:
    $$
    A e^{i(\\omega t + \\phi)}.
    $$
    """

    amplitude: CastMathExpr
    frequency: CastMathExpr
    phase: CastMathExpr


def ConstantCoefficient(value):
    """
    Function to create a constant coefficient.
    """
    return WaveCoefficient(amplitude=value, frequency=0, phase=0)


class CoefficientAdd(Coefficient):
    """
    Class representing the addition of coefficients
    """

    coeff1: CoefficientSubTypes
    coeff2: CoefficientSubTypes


class CoefficientMul(Coefficient):
    """
    Class representing the multiplication of coefficients
    """

    coeff1: CoefficientSubTypes
    coeff2: CoefficientSubTypes


########################################################################################


class Operator(TypeReflectBaseModel):
    """
    Class representing a quantum operator.
    """

    def __neg__(self):
        return OperatorScalarMul(
            op=self, coeff=WaveCoefficient(amplitude=-1, frequency=0, phase=0)
        )

    def __pos__(self):
        return self

    def __add__(self, other):
        return OperatorAdd(op1=self, op2=other)

    def __sub__(self, other):
        return OperatorAdd(
            op1=self,
            op2=OperatorScalarMul(
                op=other, coeff=WaveCoefficient(amplitude=-1, frequency=0, phase=0)
            ),
        )

    def __matmul__(self, other):
        if isinstance(other, Coefficient):
            raise TypeError(
                "Tried Kron product between Operator and Coefficient. "
                + "Scalar multiplication of Coefficient and Operator should be bracketed when perfoming Kron product."
            )
        return OperatorKron(op1=self, op2=other)

    def __mul__(self, other):
        if isinstance(other, Operator):
            return OperatorMul(op1=self, op2=other)
        else:
            return OperatorScalarMul(op=self, coeff=other)

    def __rmul__(self, other):
        return self * other

    pass


########################################################################################


def issubsystem(value: str):
    subsystem_type = value[0]
    subsystem_index = value[1:]

    if subsystem_type not in ["E", "P"]:
        raise ValueError()

    if not subsystem_index.isdigit():
        raise ValueError()

    return value


class OperatorLeaf(Operator):
    """
    Class representing a leaf operator

    Attributes:
        subsystem (Annotated[str, AfterValidator(issubsystem)]): Label for the subsystem the operator acts on.
    """

    subsystem: Annotated[str, AfterValidator(issubsystem)]


class KetBra(OperatorLeaf):
    """
    Class representing a transition operator:
    $$
    |i \\rangle \\langle j|
    $$

    Attributes:
        ket (int): End state.
        bra (int): Start state.

    """

    ket: int
    bra: int


class Annihilation(OperatorLeaf):
    """
    Class representing an annihilation operator:
    $$
    \\hat{a}
    $$
    """

    pass


class Creation(OperatorLeaf):
    """
    Class representing an annihilation operator:
    $$
    \\hat{a}^{\\dagger}
    $$
    """

    pass


class Displacement(OperatorLeaf):
    """
    Class representing a displacement operator:
    $$
    e^{\\alpha \\hat{a}^{\\dagger} - \\alpha^* \\hat{a}}
    $$

    Attributes:
        alpha (Coefficient): Displacement angle.
    """

    alpha: CoefficientSubTypes


class Identity(OperatorLeaf):
    """
    Class representing an identity operator:
    $$
    \\hat{\\mathbb{I}}
    $$
    """

    pass


class PrunedOperator(Operator):
    pass


########################################################################################


class OperatorAdd(Operator):
    """
    Class representing the addtition of [`Operators`][oqd_trical.light_matter.interface.operator.Operator].

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_trical.light_matter.interface.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_trical.light_matter.interface.operator.Operator]
    """

    op1: OperatorSubTypes
    op2: OperatorSubTypes


class OperatorMul(Operator):
    """
    Class representing the multiplication of [`Operators`][oqd_trical.light_matter.interface.operator.Operator].

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_trical.light_matter.interface.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_trical.light_matter.interface.operator.Operator]
    """

    op1: OperatorSubTypes
    op2: OperatorSubTypes


class OperatorKron(Operator):
    """
    Class representing the tensor product of [`Operators`][oqd_trical.light_matter.interface.operator.Operator].

    Attributes:
        op1 (Operator): Left hand side [`Operator`][oqd_trical.light_matter.interface.operator.Operator]
        op2 (Operator): Right hand side [`Operator`][oqd_trical.light_matter.interface.operator.Operator]
    """

    op1: OperatorSubTypes
    op2: OperatorSubTypes


class OperatorScalarMul(Operator):
    """
    Class representing the scalar multiplication of a [`Coefficient`][oqd_trical.light_matter.interface.operator.Coefficient]
    and an [`Operator`][oqd_trical.light_matter.interface.operator.Operator].

    Attributes:
        op (Operator): [`Operator`][oqd_trical.light_matter.interface.operator.Operator] to multiply.
        coeff (Coefficient): [`Coefficient`][oqd_trical.light_matter.interface.operator.Coefficient] to multiply.
    """

    op: OperatorSubTypes
    coeff: CoefficientSubTypes


########################################################################################

OperatorSubTypes = Union[
    KetBra,
    Annihilation,
    Creation,
    Identity,
    PrunedOperator,
    Displacement,
    OperatorAdd,
    OperatorMul,
    OperatorKron,
    OperatorScalarMul,
]

CoefficientSubTypes = Union[WaveCoefficient, CoefficientAdd, CoefficientMul]
