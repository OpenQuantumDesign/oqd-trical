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
import numpy as np

from oqd_compiler_infrastructure import Chain, FixedPoint, Post, Pre, RewriteRule
from oqd_core.compiler.math.passes import simplify_math_expr
from oqd_core.compiler.math.rules import (
    DistributeMathExpr,
    PartitionMathExpr,
    ProperOrderMathExpr,
)
from oqd_core.interface.math import MathNum, MathVar, MathFunc, MathExpr, MathSub, MathMul
from oqd_core.interface.atomic.protocol import SequentialProtocol
from oqd_trical.light_matter.interface.emulator import AtomicEmulatorGate

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


class _CombineTerms(RewriteRule):
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
        combiner = _CombineTerms()
        Pre(combiner)(model.base)

        return model.__class__(base=combiner.emit(), sequence=model.sequence)

    def map_AtomicEmulatorGate(self, model):
        combiner = _CombineTerms()
        Pre(combiner)(model)

        return model.__class__(hamiltonian=combiner.emit(), duration=model.duration)


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
    )

########################################################################################

class VariableSubstitution(RewriteRule):
    def __init__(self, t_offset):
        """
        Parameters:
            t_offset (str or numeric): The name or value of the time offset
                (for example, "t1") to subtract from the global time variable
        """
        self.t_offset = t_offset

    def map_MathVar(self, math_var):
        """
        If the math var is the time var 't', return an expression
        representing (t - t_offset); otherwise, return the var unchanged
        """
        if math_var.name == "t":
            return MathSub(
                expr1=MathVar(name="t"),
                # If t_offset is also a string:
                expr2=MathVar(name=self.t_offset)  
                # If t_offset is numeric change to: MathNum(value=self.t_offset)
            )
        return math_var
    
    def map_WaveCoefficient(self, wave_coeff):
        """
        Apply time-shift inside amplitude, frequency, and phase.
        Also add a constant phase offset if the frequency is a constant
        """
        # 1) Recursively rewrite the amplitude, frequency, and phase to shift 't'.
        wave_coeff.amplitude = self.apply(wave_coeff.amplitude)
        wave_coeff.frequency = self.apply(wave_coeff.frequency)
        wave_coeff.phase     = self.apply(wave_coeff.phase)

        # 2) If the frequency is a constant: add phase offset = freq * t_offset.
        #   approximate fix so that e^{i*freq*t} becomes
        #   e^{i*freq*(t - t_offset)} * e^{i*freq*t_offset} => 
        #   the second factor is the new phase offset. 
        #   If wave_coeff.frequency is a simple MathNum(...) or an integer/float 
        #   (after 'apply') we can do:
        if isinstance(wave_coeff.frequency, MathNum):
            freq_val = wave_coeff.frequency.value
            if isinstance(self.t_offset, (int, float)):
                # numeric offset => add freq_val * t_offset to the phase
                wave_coeff.phase = wave_coeff.phase + MathNum(value=freq_val * self.t_offset)
            elif isinstance(self.t_offset, str):
                # offset is symbolic => build an expression freq_val * (symbol)
                wave_coeff.phase = wave_coeff.phase + MathMul(
                    expr1=wave_coeff.frequency, 
                    expr2=MathVar(name=self.t_offset)
                )
            # else: is t_offset is more complicated than this?
        return wave_coeff

class UnfoldSequential(RewriteRule):
    def map_SequentialProtocol(self, model, operands):
        """
        Given a sequential protocol method 'unfolds' into a single parallel (time‐dependent) operator.
        
        First it must have no nested SequentialProtocol within the sequence.
        Then it computes the maximum duration (assumed to be the total protocol duration)
        and “gates” each pulse operator with a Heaviside function if its duration is shorter.
        
        Returns:
            An AtomicEmulatorGate with a combined Hamiltonian and overall duration.
        """
        # Checking for unsupported nested sequential protocols?
        for p in model.sequence:
            if isinstance(p, SequentialProtocol):
                raise NotImplementedError(
                    "SequentialProtocol within ParallelProtocol currently unsupported"
                )

        # Maximum duration among the pulses
        duration_max = np.max([p.duration for p in operands["sequence"]])

        ops = []
        for p in operands["sequence"]:
            if p.duration != duration_max:
                # Multiply the Hamiltonian by a Heaviside function that “gates” the pulse.
                # Making the Heaviside function based on:
                #     heaviside( p.duration - t )
                # which will be 1 when t < p.duration and 0 otherwise.
                ops.append(
                    p.hamiltonian * WaveCoefficient(
                        amplitude=MathFunc(
                            func="heaviside",
                            expr=p.duration - MathVar(name="t")
                        ),
                        frequency=0,
                        phase=0,
                    )
                )
            else:
                ops.append(p.hamiltonian)

        # Combining pulse operators into one overall operator
        combined_operator = reduce(lambda x, y: x + y, ops)
        return AtomicEmulatorGate(hamiltonian=combined_operator, duration=duration_max)



