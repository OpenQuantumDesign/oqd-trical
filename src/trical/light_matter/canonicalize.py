from oqd_compiler_infrastructure import RewriteRule, Pre

from oqd_core.interface.math import MathNum

from functools import reduce

########################################################################################

from .interface.operator import (
    Zero,
    Identity,
    WaveCoefficient,
    OperatorAdd,
    OperatorScalarMul,
    OperatorKron,
    OperatorMul,
)

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


########################################################################################


class OperatorDistributivity(RewriteRule):
    def map_OperatorMul(self, model):
        if isinstance(model.op1, (OperatorAdd)):
            return model.op1.__class__(
                op1=OperatorMul(op1=model.op1.op1, op2=model.op2),
                op2=OperatorMul(op1=model.op1.op2, op2=model.op2),
            )
        if isinstance(model.op2, (OperatorAdd)):
            return model.op2.__class__(
                op1=OperatorMul(op1=model.op1, op2=model.op2.op1),
                op2=OperatorMul(op1=model.op1, op2=model.op2.op2),
            )
        if isinstance(model.op1, (OperatorKron)) and isinstance(
            model.op2, (OperatorKron)
        ):
            return OperatorKron(
                op1=OperatorMul(op1=model.op1.op1, op2=model.op2.op1),
                op2=OperatorMul(op1=model.op1.op2, op2=model.op2.op2),
            )
        return None

    def map_OperatorKron(self, model):
        if isinstance(model.op1, (OperatorAdd)):
            return model.op1.__class__(
                op1=OperatorKron(op1=model.op1.op1, op2=model.op2),
                op2=OperatorKron(op1=model.op1.op2, op2=model.op2),
            )
        if isinstance(model.op2, (OperatorAdd)):
            return model.op2.__class__(
                op1=OperatorKron(op1=model.op1, op2=model.op2.op1),
                op2=OperatorKron(op1=model.op1, op2=model.op2.op2),
            )
        return None

    def map_OperatorScalarMul(self, model):
        if isinstance(model.op, (OperatorAdd)):
            return model.op.__class__(
                op1=OperatorScalarMul(op=model.op.op1, coeff=model.coeff),
                op2=OperatorScalarMul(op=model.op.op2, coeff=model.coeff),
            )
        return None

    pass


class OperatorAssociativity(RewriteRule):
    def map_OperatorAdd(self, model):
        return self._map_addmullkron(model=model)

    def map_OperatorMul(self, model):
        return self._map_addmullkron(model=model)

    def map_OperatorKron(self, model):
        return self._map_addmullkron(model=model)

    def _map_addmullkron(self, model):
        if isinstance(model.op2, model.__class__):
            return model.__class__(
                op1=model.__class__(op1=model.op1, op2=model.op2.op1),
                op2=model.op2.op2,
            )
        return model.__class__(op1=model.op1, op2=model.op2)


class GatherCoefficient(RewriteRule):
    def map_OperatorScalarMul(self, model):
        if isinstance(model.op, OperatorScalarMul):
            return (model.coeff * model.op.coeff) * model.op.op

    def map_OperatorMul(self, model):
        return self._map_mulkron(model)

    def map_OperatorKron(self, model):
        return self._map_mulkron(model)

    def _map_mulkron(self, model):
        if isinstance(model.op1, OperatorScalarMul) and isinstance(
            model.op2, OperatorScalarMul
        ):
            return (
                model.op1.coeff
                * model.op2.coeff
                * model.__class__(op1=model.op1.op, op2=model.op2.op)
            )
        if isinstance(model.op1, OperatorScalarMul):
            return model.op1.coeff * model.__class__(op1=model.op1.op, op2=model.op2)

        if isinstance(model.op2, OperatorScalarMul):
            return model.op2.coeff * model.__class__(op1=model.op1, op2=model.op2.op)


########################################################################################


class _CombineTerms(RewriteRule):
    def __init__(self):
        super().__init__()

        self.operators = []

    @property
    def coefficients(self):
        return [o[0] for o in self.operators]

    @property
    def terms(self):
        return [o[1] for o in self.operators]

    def emit(self):
        return reduce(
            lambda op1, op2: op1 + op2,
            [o[0] * o[1] for o in reversed(self.operators)],
        )

    def map_OperatorAdd(self, model):
        if model.op2.op in self.terms:
            i = self.terms.index(model.op2.op)
            self.operators[i] = (model.op2.coeff + self.coefficients[i], model.op2.op)
        else:
            self.operators.append((model.op2.coeff, model.op2.op))

        if isinstance(model.op1, OperatorAdd):
            return

        if model.op1.op in self.terms:
            i = self.terms.index(model.op1.op)
            self.operators[i] = (model.op1.coeff + self.coefficients[i], model.op1.op)
        else:
            self.operators.append((model.op1.coeff, model.op1.op))


class CombineTerms(RewriteRule):
    def map_AtomicEmulatorCircuit(self, model):
        combiner = _CombineTerms()
        Pre(combiner)(model.base)

        return model.__class__(base=combiner.emit(), sequence=model.sequence)

    def map_AtomicEmulatorGate(self, model):
        combiner = _CombineTerms()
        Pre(combiner)(model)
        return model.__class__(hamiltonian=combiner.emit(), duration=model.duration)
