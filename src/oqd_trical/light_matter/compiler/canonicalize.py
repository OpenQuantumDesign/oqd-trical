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

from functools import reduce
from typing import Union

from oqd_compiler_infrastructure import Chain, FixedPoint, Post, Pre, RewriteRule
from oqd_core.compiler.math.passes import simplify_math_expr
from oqd_core.compiler.math.rules import (
    DistributeMathExpr,
    PartitionMathExpr,
    ProperOrderMathExpr,
)
from oqd_core.interface.math import MathNum

########################################################################################
from oqd_trical.light_matter.interface.operator import (
    ConstantCoefficient,
    Identity,
    OperatorAdd,
    OperatorKron,
    OperatorMul,
    OperatorScalarMul,
    PrunedOperator,
    WaveCoefficient,
)

########################################################################################


class Prune(RewriteRule):
    """Prunes an Operator AST by removing zeros"""

    def map_OperatorAdd(self, model):
        if isinstance(model.op1, PrunedOperator):
            return model.op2
        if isinstance(model.op2, PrunedOperator):
            return model.op1

    def map_OperatorMul(self, model):
        if isinstance(model.op1, PrunedOperator) or isinstance(
            model.op2, PrunedOperator
        ):
            return PrunedOperator()

    def map_OperatorScalarMul(self, model):
        if isinstance(
            model.coeff, WaveCoefficient
        ) and model.coeff.amplitude == MathNum(value=0):
            return PrunedOperator()
        if isinstance(model.op, PrunedOperator):
            return PrunedOperator()

    def map_OperatorKron(self, model):
        if isinstance(model.op1, PrunedOperator) or isinstance(
            model.op2, PrunedOperator
        ):
            return PrunedOperator()

    def map_CoefficientAdd(self, model):
        if isinstance(
            model.coeff1, WaveCoefficient
        ) and model.coeff1.amplitude == MathNum(value=0):
            return model.coeff2
        if isinstance(
            model.coeff2, WaveCoefficient
        ) and model.coeff2.amplitude == MathNum(value=0):
            return model.coeff1

    def map_Displacement(self, model):
        if isinstance(
            model.alpha, WaveCoefficient
        ) and model.alpha.amplitude == MathNum(value=0):
            return Identity(subsystem=model.subsystem)


########################################################################################


class PruneZeroPowers(RewriteRule):
    """Prunes a MathExpr AST by MathPow when base is zero"""

    def map_MathPow(self, model):
        if model.expr1 == MathNum(value=0):
            return MathNum(value=0)


########################################################################################


class OperatorDistributivity(RewriteRule):
    """Implements distributivity of addition over multiplication, kronecker product and scalar multiplication"""

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
    """Implements associativity of addition, multiplication and kronecker product"""

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
    """Gathers the coefficients of an operator into a single scalar multiplication for each term"""

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


class CombineCoefficient(RewriteRule):
    def map_CoefficientMul(self, model):
        if isinstance(model.coeff1, WaveCoefficient) and isinstance(
            model.coeff2, WaveCoefficient
        ):
            return WaveCoefficient(
                amplitude=model.coeff1.amplitude * model.coeff2.amplitude,
                frequency=model.coeff1.frequency + model.coeff2.frequency,
                phase=model.coeff1.phase + model.coeff2.phase,
            )

    def map_CoefficientAdd(self, model):
        if (
            isinstance(model.coeff1, WaveCoefficient)
            and isinstance(model.coeff2, WaveCoefficient)
            and model.coeff1.frequency == MathNum(value=0)
            and model.coeff1.phase == MathNum(value=0)
            and model.coeff2.frequency == MathNum(value=0)
            and model.coeff2.phase == MathNum(value=0)
        ):
            return ConstantCoefficient(
                value=model.coeff1.amplitude + model.coeff2.amplitude
            )


class ScaleTerms(RewriteRule):
    """Adds a scalar multiplication for each term if it does not exist"""

    def __init__(self):
        super().__init__()
        self.op_add_root = False

    def map_AtomicEmulatorGate(self, model):
        self.op_add_root = False

    def map_OperatorAdd(self, model):
        self.op_add_root = True
        if isinstance(model.op1, Union[OperatorAdd, OperatorScalarMul]):
            op1 = model.op1
        else:
            op1 = ConstantCoefficient(value=1) * model.op1
        if isinstance(model.op2, OperatorScalarMul):
            op2 = model.op2
        else:
            op2 = ConstantCoefficient(value=1) * model.op2
        return op1 + op2

    def map_OperatorScalarMul(self, model):
        self.op_add_root = True
        pass

    def map_Operator(self, model):
        if not self.op_add_root:
            self.op_add_root = True
            return ConstantCoefficient(value=1) * model


########################################################################################


class _CombineTermsHelper(RewriteRule):
    """Helper for combining terms of the same operator by combining their coefficients"""

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
        if self.operators == []:
            return PrunedOperator()
        return reduce(
            lambda op1, op2: op1 + op2,
            [o[0] * o[1] for o in self.operators],
        )

    def map_OperatorScalarMul(self, model):
        if model.op in self.terms:
            i = self.terms.index(model.op)
            self.operators[i] = (model.coeff + self.coefficients[i], model.op)
        else:
            self.operators.append((model.coeff, model.op))


class CombineTerms(RewriteRule):
    """Combines terms of the same operator by combining their coefficients"""

    def map_AtomicEmulatorCircuit(self, model):
        combiner = _CombineTermsHelper()
        Pre(combiner)(model.base)

        return model.__class__(
            frame=model.frame, base=combiner.emit(), sequence=model.sequence
        )

    def map_AtomicEmulatorGate(self, model):
        combiner = _CombineTermsHelper()
        Pre(combiner)(model)

        return model.__class__(hamiltonian=combiner.emit(), duration=model.duration)


class RelabelStates(RewriteRule):
    def __init__(self, relabel_rules):
        self._relabel_rules = relabel_rules

    def map_KetBra(self, model):
        new_ket = self._relabel_rules[model.subsystem][model.ket]
        new_bra = self._relabel_rules[model.subsystem][model.bra]

        return model.__class__(ket=new_ket, bra=new_bra, subsystem=model.subsystem)


########################################################################################
class PushBaseHamiltonian(RewriteRule):
    def map_AtomicEmulatorCircuit(self, model):
        self.base = model.base
        return model.__class__(
            frame=model.frame, base=PrunedOperator(), sequence=model.sequence
        )

    def map_AtomicEmulatorGate(self, model):
        return model.__class__(
            hamiltonian=model.hamiltonian + self.base, duration=model.duration
        )


########################################################################################


def canonicalization_pass_factory():
    """Creates a new instance of the canonicalization pass"""
    return Chain(
        Chain(
            FixedPoint(Post(DistributeMathExpr())),
            FixedPoint(Post(ProperOrderMathExpr())),
            FixedPoint(Post(PartitionMathExpr())),
            FixedPoint(Post(PruneZeroPowers())),
        ),
        simplify_math_expr,
        FixedPoint(Post(Prune())),
        Pre(PushBaseHamiltonian()),
        Chain(
            FixedPoint(Post(OperatorDistributivity())),
            FixedPoint(Post(OperatorAssociativity())),
            Post(GatherCoefficient()),
            FixedPoint(Post(CombineCoefficient())),
        ),
        Pre(ScaleTerms()),
        Post(CombineTerms()),
        Chain(
            FixedPoint(Pre(DistributeMathExpr())),
            FixedPoint(Post(ProperOrderMathExpr())),
            FixedPoint(Post(PartitionMathExpr())),
            FixedPoint(Post(PruneZeroPowers())),
        ),
        simplify_math_expr,
        FixedPoint(Post(Prune())),
    )
