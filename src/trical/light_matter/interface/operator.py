from typing import Union

from midstack.interface.base import TypeReflectBaseModel

########################################################################################

from .coefficient import Coefficient, ConstantCoefficient, WaveCoefficient

########################################################################################


class Operator(TypeReflectBaseModel):
    """Class specifying operations of Operator-type objects"""

    def __neg__(self):
        return OperatorScalarMul(op=self, coeff=ConstantCoefficient(value=-1))

    def __pos__(self):
        return self

    def __add__(self, other):

        return OperatorAdd(op1=self, op2=other)

    def __sub__(self, other):

        return OperatorAdd(
            op1=self,
            op2=OperatorScalarMul(op=other, coeff=ConstantCoefficient(value=-1)),
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


class KetBra(Operator):
    """Class specifying the projector onto ion (internal) state $|i \\rangle \\langle j|$

    Attributes:
        i (int): ket index
        j (int): bra index
        dims (int):  dimensions of ion Hilbert space (number of ions)
    """

    i: int
    j: int
    dims: int


class Annihilation(Operator):
    dims: int
    pass


class Creation(Operator):
    dims: int
    pass


class Identity(Operator):
    dims: int
    pass


class Zero(Operator):
    pass


class hc(Operator):
    """Class for object representing a pending hermitian conjugation"""

    pass


class Displacement(Operator):
    """Class representing the displacement operator: $D(\\alpha) = e^{\\alpha a^{\\dagger} - \\alpha^* a}$

    Attributes:
        alpha (Coefficient): most often a WaveCofficient since coherent state parameter is a function of time
        dims (int): size of motional mode Hilbert space (maximum number of phonon modes for single mode)
    """

    alpha: Coefficient
    dims: int
    pass


class ApproxDisplacementMatrix(Operator):
    """Class storing all the argument needs to produce an approximated displacement operator when converting Hamiltonian tree to QobjEvo

    Attributes:
        ld_order (int): order to which to take Lamb-Dicke approximation
        alpha (WaveCoefficient): time-dependent coherent state parameters
        rwa_cutoff (Union[float, str]): remove all terms rotating faster than rwa_cutoff. Allowable str is 'inf'
        Delta (float): detuning in the prefactor term for both the internal and motional coupling
        nu (float): eigenmode frequency
        dims (int): size of motional mode Hilbert space (maximum number of phonon modes for single mode)
    """

    ld_order: int
    alpha: WaveCoefficient
    rwa_cutoff: Union[float, str]
    Delta: float
    nu: float
    dims: int


class OperatorAdd(Operator):
    op1: Operator
    op2: Operator


class OperatorMul(Operator):
    op1: Operator
    op2: Operator


class OperatorKron(Operator):
    op1: Operator
    op2: Operator


class OperatorScalarMul(Operator):
    op: Operator
    coeff: Coefficient
