from cmath import exp

from qutip import qeye, basis, tensor, destroy, QobjEvo

from midstack.compiler import *

########################################################################################

from ...light_matter.utilities import displace

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
