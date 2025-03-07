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

from __future__ import annotations

from typing import Annotated, Union

from oqd_compiler_infrastructure import Post, RewriteRule, TypeReflectBaseModel
from oqd_core.interface.math import CastMathExpr, MathFunc
from pydantic import AfterValidator

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
        return self + (-other)

    def __mul__(self, other):
        if isinstance(other, Coefficient):
            return CoefficientMul(coeff1=self, coeff2=other)
        else:
            return other * self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(self, WaveCoefficient) and isinstance(other, WaveCoefficient):
            return CoefficientMul(
                coeff1=self,
                coeff2=WaveCoefficient(
                    amplitude=1 / other.amplitude,
                    frequency=-other.frequency,
                    phase=-other.phase,
                ),
            )
        if isinstance(self, CoefficientAdd) and isinstance(other, WaveCoefficient):
            return CoefficientAdd(
                coeff1=self.coeff1 / other, coeff2=self.coeff2 / other
            )
        if isinstance(self, CoefficientMul) and isinstance(other, WaveCoefficient):
            return CoefficientMul(coeff1=self.coeff1, coeff2=self.coeff2 / other)
        else:
            raise TypeError("Division only supported for WaveCoefficients denominator")

    def conj(self):
        return Post(ConjugateCoefficient())(self)

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


class ConjugateCoefficient(RewriteRule):
    def map_WaveCoefficient(self, model):
        return model.__class__(
            amplitude=MathFunc(func="conj", expr=model.amplitude),
            frequency=-MathFunc(func="conj", expr=model.frequency),
            phase=-MathFunc(func="conj", expr=model.phase),
        )


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


def Number(*, subsystem):
    return Creation(subsystem=subsystem) * Annihilation(subsystem=subsystem)


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
