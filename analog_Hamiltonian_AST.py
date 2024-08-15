from qutip import qeye, basis, tensor, destroy, QobjEvo
from cmath import exp

########################################################################################

from quantumion.interface.base import TypeReflectBaseModel
from quantumion.compilerv2 import *

########################################################################################
from utilities import displace

########################################################################################


class Coefficient(TypeReflectBaseModel):
    def __neg__(self):
        return CoefficientMul(coeff1=self, coeff2=ConstantCoefficient(value=-1))

    def __pos__(self):
        return self


    def __add__(self, other):
        return CoefficientAdd(coeff1=self, coeff2=other)

    def __sub__(self, other):
        return CoefficientAdd(
            coeff1=self,
            coeff2=CoefficientMul(coeff1=other, coeff2=ConstantCoefficient(value=-1)),
        )

    def __mul__(self, other):

        if isinstance(other, Operator):
            return OperatorScalarMul(op = other,  coeff= self)

        return CoefficientMul(coeff1=self, coeff2=other)
    

    pass


class ConstantCoefficient(Coefficient):
    value: float


class WaveCoefficient(Coefficient):
    amplitude: float
    frequency: float
    phase: float
    ion_indx: int | None
    laser_indx: int | None
    mode_indx: int | None
    i: int | None
    j: int | None

class CoefficientAdd(Coefficient):
    coeff1: Coefficient
    coeff2: Coefficient


class CoefficientMul(Coefficient):
    coeff1: Coefficient
    coeff2: Coefficient

########################################################################################


class Operator(TypeReflectBaseModel):
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

    pass


class KetBra(Operator):
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
    pass


class Displacement(Operator):
    alpha: Coefficient
    dims: int
    pass
    

class ApproxDisplacementMatrix(Operator):
    ld_order: int
    alpha: WaveCoefficient
    rwa_cutoff: float | str # a float or 'inf'
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

########################################################################################

class QutipConversion(RewriteRule):
    def __init__(self, ld_order = 1):
        super().__init__()
        self.operands = []
        self.ld_order = ld_order

    def map_ApproxDisplacementMatrix(self, model):
        
        alpha = self.operands.pop()
        

        op = displace(ld_order = model.ld_order,
                          alpha = alpha,
                          rwa_cutoff = model.rwa_cutoff,
                          Delta = model.Delta, 
                          nu = model.nu,
                          dims = model.dims)
        
        self.operands.append(op)

    def map_KetBra(self, model):

        op = basis(model.dims, model.dims - model.i - 1) * basis(model.dims, model.dims - model.j - 1).dag()
        self.operands.append(op)
        
        pass

    def map_Annihilation(self, model):
        op = destroy(model.dims)
        self.operands.append(op)
        pass

    def map_Creation(self, model):
        op = destroy(model.dims).dag()
        self.operands.append(op)     
        pass

    def map_Identity(self, model):
        op = qeye(model.dims)
        self.operands.append(op)
        pass



    def map_OperatorAdd(self, model):

        if len(self.operands) == 1:
            # Case where we have an op and its hc

            op1 = self.operands.pop()

            def time_dep_fn(t, args = {}):
                return op1(t, args) + op1(t, args).dag()
            
            op = QobjEvo(time_dep_fn)
        else:


            op2 = self.operands.pop()
            op1 = self.operands.pop()

            if isinstance(op1, QobjEvo) and isinstance(op2, QobjEvo):
                
                def time_dep_fn(t, args = {}):
                    return op1(t, args) + op2(t, args)
                
                op = QobjEvo(time_dep_fn)
            else:
                op = op1 + op2
            
        self.operands.append(op)
        pass

    def map_OperatorMul(self, model):
        op2 = self.operands.pop()
        op1 = self.operands.pop()


        if isinstance(op1, QobjEvo) and isinstance(op2, QobjEvo):
            def time_dep_fn(t, args = {}):
                return op1(t, args) * op2(t, args)
            op = QobjEvo(time_dep_fn)
        else:

            op = op1 @ op2  
        self.operands.append(op)
        pass

    def map_OperatorKron(self, model):

        op2 = self.operands.pop()
        op1 = self.operands.pop()
        op = tensor(op1, op2)

        self.operands.append(op)
        
        pass

    def map_OperatorScalarMul(self, model):
        coeff = self.operands.pop()
        op = self.operands.pop()

        if callable(coeff): # map returned some function of t and args

            time_dep_fn = lambda t, args = {}: coeff(t,args) * qeye(op.dims[1])
            coeffp = QobjEvo(time_dep_fn)
        else:
            coeffp = coeff

        self.operands.append(coeffp * op)
        pass

    def map_CoefficientAdd(self, model):
        coeff2 = self.operands.pop()
        coeff1 = self.operands.pop()


        if callable(coeff1) and not callable(coeff2):
            def time_dep_fn(t, args = {}):
                return coeff1(t, args) + coeff2
            op = time_dep_fn

        elif callable(coeff2) and not callable(coeff1):
            def time_dep_fn(t, args = {}):
                return coeff1 + coeff2(t, args)
            op = time_dep_fn
        elif  callable(coeff1) and callable(coeff2):
            def time_dep_fn(t, args = {}):
                return coeff1(t, args) + coeff2(t, args)

            op = time_dep_fn
        else:
            op = coeff1 + coeff2
            
        self.operands.append(op)
        pass

    def map_CoefficientMul(self, model):
        coeff2 = self.operands.pop()
        coeff1 = self.operands.pop()

        if callable(coeff1) and not callable(coeff2):
            def time_dep_fn(t, args = {}):
                return coeff1(t, args) * coeff2

            op = time_dep_fn

        elif callable(coeff2) and not callable(coeff1):
            def time_dep_fn(t, args = {}):
                return coeff1 * coeff2(t, args)
            op = time_dep_fn
        elif  callable(coeff1) and callable(coeff2):
            def time_dep_fn(t, args = {}):
                return coeff1(t, args) * coeff2(t, args)

            op = time_dep_fn
        else:
            op = coeff1 * coeff2
            
        self.operands.append(op)
        pass

    def map_ConstantCoefficient(self, model):
        self.operands.append(model.value)
        pass


    def map_WaveCoefficient(self, model):        
        A = model.amplitude
        omega = model.frequency
        phi = model.phase

        time_dep_fn = lambda t, args={}: A * exp(1j * (omega * t + phi))
        
        self.operands.append(time_dep_fn)
        
        pass