from qutip import qeye, basis, tensor, destroy, QobjEvo
from cmath import exp

########################################################################################

from oqd_compiler_infrastructure import TypeReflectBaseModel, RewriteRule

########################################################################################
from trical.light_matter.utilities import displace

########################################################################################


class Coefficient(TypeReflectBaseModel):
    """Class specifying operations of Coefficient-type objects"""

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
            return OperatorScalarMul(op=other, coeff=self)

        return CoefficientMul(coeff1=self, coeff2=other)

    pass


class ConstantCoefficient(Coefficient):
    """Class for coefficients that do not change in time"""

    value: float


class WaveCoefficient(Coefficient):
    """Class for coefficients of the form A\exp(i*(\omega*t + \phi))

    Attributes:
        amplitude (float): A
        frequency (float): \omega
        phase (float): \phi
        ion_indx (int): identification index of the ion as defined when instantiating the ion
        laser_indx (int): identification index of the laser as defined when instantiating the laser
        mode_indx (int): identification index of the mode as defined when instantiating the mode
        i (int): i in |iXj| if WaveCoefficient is a prefactor for a KetBra
        j (int): j in |iXj| if WaveCoefficient is a prefactor for a KetBra
    """

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
    """Class specifying the projector onto ion (internal) state |iXj|

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
    """Class representing the displacement operator: D(\alpha) = \exp(\alpha a^{\dag} - \alpha^* a)

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
        rwa_cutoff (float or str): remove all terms rotating faster than rwa_cutoff. Allowable str is 'inf'
        Delta (float): detuning in the prefactor term for both the internal and motional coupling
        nu (float): eigenmode frequency
        dims (int): size of motional mode Hilbert space (maximum number of phonon modes for single mode)
    """

    ld_order: int
    alpha: WaveCoefficient
    rwa_cutoff: float | str  # a float or 'inf'
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
    """ReWrite rule for converting a Hamiltonian tree object to a time-dependent QuTiP object (QobjEvo)

    Attributes:
        operands (list): operand stack used for converting between representations

    """

    def __init__(self, ld_order=1):
        super().__init__()
        self.operands = []

    def map_ApproxDisplacementMatrix(self, model):
        alpha = self.operands.pop()

        op = displace(
            ld_order=model.ld_order,
            alpha=alpha,
            rwa_cutoff=model.rwa_cutoff,
            Delta=model.Delta,
            nu=model.nu,
            dims=model.dims,
        )

        self.operands.append(op)

    def map_KetBra(self, model):
        """Function mapping KetBra objects to a Qutip projector
        Attributes:
            model (KetBra): KetBra object |iXj|
        """

        op = (
            basis(model.dims, model.dims - model.i - 1)
            * basis(model.dims, model.dims - model.j - 1).dag()
        )
        self.operands.append(op)

        pass

    def map_Annihilation(self, model):
        """Function mapping Annhilation objects to a Qutip destroy object

        Attributes:
            model (Annhilation):
        """
        op = destroy(model.dims)
        self.operands.append(op)
        pass

    def map_Creation(self, model):
        """Function mapping Creation objects to a Qutip destroy.dag() object

        Attributes:
            model (Creation):
        """
        op = destroy(model.dims).dag()
        self.operands.append(op)
        pass

    def map_Identity(self, model):
        """Function mapping Identity objects to a Qutip qeye() object

        Attributes:
            model (Identity):
        """
        op = qeye(model.dims)
        self.operands.append(op)
        pass

    def map_OperatorAdd(self, model):
        if len(self.operands) == 1:
            # Case where we have an op and its hc

            op1 = self.operands.pop()

            def time_dep_fn(t, args={}):
                return op1(t, args) + op1(t, args).dag()

            op = QobjEvo(time_dep_fn)
        else:
            op2 = self.operands.pop()
            op1 = self.operands.pop()

            if isinstance(op1, QobjEvo) and isinstance(op2, QobjEvo):

                def time_dep_fn(t, args={}):
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

            def time_dep_fn(t, args={}):
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

        if callable(coeff):  # map returned some function of t and args
            time_dep_fn = lambda t, args={}: coeff(t, args) * qeye(op.dims[1])
            coeffp = QobjEvo(time_dep_fn)
        else:
            coeffp = coeff

        self.operands.append(coeffp * op)
        pass

    def map_CoefficientAdd(self, model):
        coeff2 = self.operands.pop()
        coeff1 = self.operands.pop()

        if callable(coeff1) and not callable(coeff2):

            def time_dep_fn(t, args={}):
                return coeff1(t, args) + coeff2

            op = time_dep_fn

        elif callable(coeff2) and not callable(coeff1):

            def time_dep_fn(t, args={}):
                return coeff1 + coeff2(t, args)

            op = time_dep_fn
        elif callable(coeff1) and callable(coeff2):

            def time_dep_fn(t, args={}):
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

            def time_dep_fn(t, args={}):
                return coeff1(t, args) * coeff2

            op = time_dep_fn

        elif callable(coeff2) and not callable(coeff1):

            def time_dep_fn(t, args={}):
                return coeff1 * coeff2(t, args)

            op = time_dep_fn
        elif callable(coeff1) and callable(coeff2):

            def time_dep_fn(t, args={}):
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
