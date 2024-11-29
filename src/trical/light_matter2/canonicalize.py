from oqd_compiler_infrastructure import RewriteRule

from oqd_core.interface.math import MathNum

########################################################################################

from .interface.operator import Zero, Identity, WaveCoefficient

########################################################################################


class Prune(RewriteRule):
    def map_OperatorAdd(self, model):
        if isinstance(model.op1, Zero):
            return model.op2
        if isinstance(model.op2, Zero):
            return model.op1

    def map_OperatorMul(self, model):
        if isinstance(model.op1, Zero) or isinstance(model.op2, Zero):
            return Zero()

    def map_OperatorScalarMul(self, model):
        if isinstance(
            model.coeff, WaveCoefficient
        ) and model.coeff.amplitude == MathNum(value=0):
            return Zero()

    def map_OperatorKron(self, model):
        if isinstance(model.op1, Zero) or isinstance(model.op2, Zero):
            return Zero()

    def map_CoefficientAdd(self, model):
        if isinstance(
            model.coeff1, WaveCoefficient
        ) and model.coeff1.amplitude == MathNum(value=0):
            return model.coeff2
        if isinstance(
            model.coeff2, WaveCoefficient
        ) and model.coeff2.amplitude == MathNum(value=0):
            return model.coeff1

    def map_Wave(self, model):
        if isinstance(
            model.lamb_dicke, WaveCoefficient
        ) and model.lamb_dicke.amplitude == MathNum(value=0):
            return Identity()


########################################################################################


class PruneZeroPowers(RewriteRule):
    def map_MathPow(self, model):
        if model.expr1 == MathNum(value=0):
            return MathNum(value=0)
