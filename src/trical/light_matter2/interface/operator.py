from __future__ import annotations
from typing import Union

from oqd_compiler_infrastructure import TypeReflectBaseModel
from oqd_core.interface.math import CastMathExpr

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


class KetBra(Operator):
    ket: int
    bra: int


class Ladder(Operator):
    pass


class Annihilation(Ladder):
    pass


class Creation(Ladder):
    pass


class Displacement(Operator):
    alpha: CoefficientSubTypes


class Identity(Operator):
    pass


class Zero(Operator):
    pass


class Wave(Operator):
    lamb_dicke: CoefficientSubTypes


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
    A cos(omega t + phi)
    """

    amplitude: CastMathExpr
    frequency: CastMathExpr
    phase: CastMathExpr


class CoefficientAdd(Coefficient):
    coeff1: CoefficientSubTypes
    coeff2: CoefficientSubTypes


class CoefficientMul(Coefficient):
    coeff1: CoefficientSubTypes
    coeff2: CoefficientSubTypes


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
    Zero,
    Wave,
    Displacement,
    OperatorAdd,
    OperatorMul,
    OperatorKron,
    OperatorScalarMul,
]

CoefficientSubTypes = Union[WaveCoefficient, CoefficientAdd, CoefficientMul]
