from typing import Union, List, Tuple
import oqd_compiler_infrastructure as ci  
from trical.light_matter.interface.coefficient import CoefficientHC
from oqd_compiler_infrastructure import TypeReflectBaseModel
from .coefficient import Coefficient, ConstantCoefficient, WaveCoefficient

class Operator(TypeReflectBaseModel):
    """Class specifying operations of Operator-type objects"""

    def __neg__(self):
        return OperatorScalarMul(op=self, coeff=ConstantCoefficient(value=-1))

    def __pos__(self):
        return self

    def __add__(self, other):
        if isinstance(other, Operator):
            return OperatorAdd(op1=self, op2=other)
        else:
            raise TypeError("Can only add Operator with another Operator.")

    def __sub__(self, other):
        if isinstance(other, Operator):
            return OperatorAdd(
                op1=self,
                op2=OperatorScalarMul(op=other, coeff=ConstantCoefficient(value=-1)),
            )
        else:
            raise TypeError("Can only subtract Operator with another Operator.")

    def __matmul__(self, other):
        if isinstance(other, Coefficient):
            raise TypeError(
                "Attempted Kron product between Operator and Coefficient. "
                "Scalar multiplication of Coefficient and Operator should be bracketed when performing Kron product."
            )
        elif isinstance(other, Operator):
            return OperatorKron(op1=self, op2=other)
        else:
            raise TypeError("Can only perform Kron product with another Operator.")

    def __mul__(self, other):
        if isinstance(other, Operator):
            return OperatorMul(op1=self, op2=other)
        elif isinstance(other, Coefficient):
            return OperatorScalarMul(op=self, coeff=other)
        else:
            raise TypeError("Can only multiply Operator with Coefficient or another Operator.")

    def __rmul__(self, other):
        return self * other

########################################################################################

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
    """Class representing the annihilation operator"""
    pass

class Creation(Operator):
    """Class representing the creation operator"""
    pass

class Identity(Operator):
    """Class representing the identity operator"""
    pass

class Zero(Operator):
    """Class representing the zero operator"""

    pass

class OperatorHCRule(ci.ConversionRule):
    def map_OperatorScalarMul(self, model, operands):
        coeff_conj = CoefficientHC()(model.coeff)
        op_conj = hc(op=model.op)
        return ci.OperatorScalarMul(coeff=coeff_conj, op=op_conj)

    def map_OperatorAdd(self, model, operands):
        ops_conj = [hc(op=op) for op in model.ops]
        return ci.OperatorAdd(*ops_conj)

    def map_OperatorMul(self, model, operands):
        ops_conj = [hc(op=op) for op in reversed(model.ops)]
        return ci.OperatorMul(*ops_conj)

    def map_KetBra(self, model, operands):
        return ci.KetBra(i=model.j, j=model.i, dims=model.dims)

    def generic_map(self, model, operands):
        return model

class hc(Operator):
    """Class representing a pending Hermitian conjugation of an operator."""
    
    op: Operator
    def apply(self):
        # Use the traversal to apply Hermitian conjugation
        rule = OperatorHCRule()
        return rule(self.op)

class Displacement(Operator):

    """Class representing the displacement operator: $D(\\alpha) = e^{\\alpha a^{\\dagger} - \\alpha^* a}$

    Attributes:
        alpha (Coefficient): Most often a WaveCoefficient since the coherent state parameter is a function of time
        dims (int): Size of motional mode Hilbert space (maximum number of phonon modes for single mode)
    """

    def __init__(self, alpha: Coefficient, dims: int):
        self.alpha = alpha
        self.dims = dims

class ApproxDisplacementMatrix(Operator):
    """Class storing all the arguments needed to produce an approximated displacement operator when converting Hamiltonian tree to QobjEvo.

    Attributes:
        ld_order (int): Order to which to take the Lamb-Dicke approximation
        alpha (WaveCoefficient): Time-dependent coherent state parameter
        rwa_cutoff (Union[float, str]): Remove all terms rotating faster than rwa_cutoff; Allowable str is 'inf'
        Delta (float): Detuning in the prefactor term for both internal and motional coupling
        nu (float): Eigenmode frequency
        dims (int): Size of motional mode Hilbert space (maximum number of phonon modes for single mode)
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

def extract_indices_dims(ekron: OperatorKron) -> Tuple[int, int, int]:
    """Extracts the indices and dimensions from nested OperatorKron structures containing a KetBra operator."""

    op1, op2 = ekron.op1, ekron.op2

    if isinstance(op1, KetBra):
        return op1.i, op1.j, op1.dims
    elif isinstance(op2, KetBra):
        return op2.i, op2.j, op2.dims

    if isinstance(op1, OperatorKron):
        return extract_indices_dims(op1)
    elif isinstance(op2, OperatorKron):
        return extract_indices_dims(op2)

def extract_alphas(ekron: OperatorKron) -> List[WaveCoefficient]:
    """Extracts the list of coherent state parameters from nested tensor product structures containing Displacement operators."""

    alphas = []
    nodes_to_visit = [ekron]

    while nodes_to_visit:
        current_node = nodes_to_visit.pop()
        if isinstance(current_node, OperatorKron):
            nodes_to_visit.append(current_node.op1)
            nodes_to_visit.append(current_node.op2)
        elif isinstance(current_node, Displacement):
            alphas.append(current_node.alpha)
    
    return alphas

def extended_kron(op_list: List[Operator]) -> OperatorKron:
    """Recursively constructs a nested tensor product (OperatorKron) from a list of operators."""
    
    if len(op_list) == 2:
        return OperatorKron(op1=op_list[0], op2=op_list[1])
    elif len(op_list) > 2:
        return OperatorKron(op1=op_list[0], op2=extended_kron(op_list[1:]))
    else:
        raise ValueError("The list must contain at least two operators for a valid tensor product.")

