from __future__ import annotations

from typing import Union, Annotated

from pydantic import AfterValidator

from oqd_compiler_infrastructure import TypeReflectBaseModel
from oqd_core.interface.math import CastMathExpr


########################################################################################


class Coefficient(TypeReflectBaseModel):
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
    A exp(i * (omega t + phi))
    """

    amplitude: CastMathExpr
    frequency: CastMathExpr
    phase: CastMathExpr


def ConstantCoefficient(value):
    return WaveCoefficient(amplitude=value, frequency=0, phase=0)


class CoefficientAdd(Coefficient):
    coeff1: CoefficientSubTypes
    coeff2: CoefficientSubTypes


class CoefficientMul(Coefficient):
    coeff1: CoefficientSubTypes
    coeff2: CoefficientSubTypes


########################################################################################


class Operator(TypeReflectBaseModel):
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
    subsystem: Annotated[str, AfterValidator(issubsystem)]


class KetBra(OperatorLeaf):
    ket: int
    bra: int


class Annihilation(OperatorLeaf):
    pass


class Creation(OperatorLeaf):
    pass


class Displacement(OperatorLeaf):
    alpha: CoefficientSubTypes


class Identity(OperatorLeaf):
    pass


class Wave(OperatorLeaf):
    lamb_dicke: CoefficientSubTypes


class PrunedOperator(Operator):
    pass


########################################################################################


class OperatorAdd(Operator):
    op1: OperatorSubTypes
    op2: OperatorSubTypes


class OperatorMul(Operator):
    op1: OperatorSubTypes
    op2: OperatorSubTypes


class OperatorKron(Operator):
    op1: OperatorSubTypes
    op2: OperatorSubTypes


class OperatorScalarMul(Operator):
    op: OperatorSubTypes
    coeff: CoefficientSubTypes


########################################################################################

OperatorSubTypes = Union[
    KetBra,
    Annihilation,
    Creation,
    Identity,
    PrunedOperator,
    Wave,
    Displacement,
    OperatorAdd,
    OperatorMul,
    OperatorKron,
    OperatorScalarMul,
]

CoefficientSubTypes = Union[WaveCoefficient, CoefficientAdd, CoefficientMul]
