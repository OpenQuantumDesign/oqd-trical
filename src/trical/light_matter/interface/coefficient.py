from math import cos, sin, sqrt, atan2, pi
from copy import deepcopy
import oqd_compiler_infrastructure as ci  
from oqd_compiler_infrastructure import TypeReflectBaseModel, RewriteRule

########################################################################################

class Coefficient(TypeReflectBaseModel):
    """Abstract base class specifying operations of Coefficient-type objects."""

    def __neg__(self):
        return CoefficientMul(coeff1=self, coeff2=ConstantCoefficient(value=-1))

    def __pos__(self):
        return self

    def __add__(self, other):
        if isinstance(other, Coefficient):
            return CoefficientAdd(coeff1=self, coeff2=other)
        else:
            raise TypeError("Can only add Coefficient with another Coefficient.")

    def __sub__(self, other):
        if isinstance(other, Coefficient):
            return CoefficientAdd(
                coeff1=self,
                coeff2=CoefficientMul(coeff1=other, coeff2=ConstantCoefficient(value=-1)),
            )
        else:
            raise TypeError("Can only subtract Coefficient with another Coefficient.")

    def __mul__(self, other):
        if isinstance(other, Coefficient):
            return CoefficientMul(coeff1=self, coeff2=other)
        else:
            raise TypeError("Can only multiply Coefficient with another Coefficient.")

    def __truediv__(self, other):
        if isinstance(other, Coefficient):
            inverse = CoefficientPow(coeff=other, power=-1)
            return CoefficientMul(coeff1=self, coeff2=inverse)
        else:
            raise TypeError("Can only divide Coefficient by another Coefficient.")

########################################################################################

class ConstantCoefficient(Coefficient):
    """Class for coefficients that do not change in time."""

    value: float

    def evaluate(self):
        """Evaluates the constant coefficient."""
        return self.value

########################################################################################

class WaveCoefficient(Coefficient):
    """Class for coefficients of the form A * exp(i * (ω * t + φ)).

    Attributes:
        amplitude (float): Amplitude of the wave A.
        frequency (float): Frequency of the wave ω.
        phase (float): Phase of the wave φ.
        ion_indx (int): Identification index of the ion.
        laser_indx (int): Identification index of the laser.
        mode_indx (int): Identification index of the mode.
        i (int): Index i in |i⟩⟨j| if applicable.
        j (int): Index j in |i⟩⟨j| if applicable.
    """

    def evaluate(self, t: float):
        """Evaluates the wave coefficient at time t."""
        return self.amplitude * complex(
            cos(self.frequency * t + self.phase),
            sin(self.frequency * t + self.phase),
        )

########################################################################################

class CoefficientAdd(Coefficient):
    """Class representing the addition of two Coefficients."""

    def evaluate(self, t: float):
        """Evaluates the sum of the two coefficients at time t."""
        return self.coeff1.evaluate(t) + self.coeff2.evaluate(t)

########################################################################################

class CoefficientMul(Coefficient):
    """Class representing the multiplication of two Coefficients."""

    def evaluate(self, t: float):
        """Evaluates the product of the two coefficients at time t."""
        return self.coeff1.evaluate(t) * self.coeff2.evaluate(t)

########################################################################################

class CoefficientPow(Coefficient):
    """Class representing a Coefficient raised to a power."""

    def evaluate(self, t: float):
        """Evaluates the coefficient raised to the given power at time t."""
        return self.coeff.evaluate(t) ** self.power

########################################################################################

class CoefficientHCRule(ci.ConversionRule):
    """Class representing Hermitian conjugate (HC) operations on coefficients."""

    def map_CoefficientAdd(self, expr, args):
        """
        Handle the Hermitian conjugate of a sum of coefficients
        (a + b)^\dagger = a^\dagger + b^\dagger
        """
        return self.apply_HC(args)

    def map_CoefficientMul(self, expr, args):
        """
        Handle the Hermitian conjugate of a product of coefficients
        (a * b)^\dagger = a^\dagger * b^\dagger
        Note: Order is reversed if coefficients are non-commutative
        """
        # Assuming coefficients are commutative if not reverse order!!! (going with reverse order for now)
        conjugated_args = [self.apply_HC(arg) for arg in reversed(args)]
        return ci.CoefficientMul(*conjugated_args)

    def map_CoefficientPow(self, expr, args):
        """
        Handle the Hermitian conjugate of a coefficient raised to a power.
        (a^n)^\dagger = (a^\dagger)^n
        """
        base, exponent = args
        conjugated_base = self.apply_HC(base)
        return ci.CoefficientPow(conjugated_base, exponent)

    def map_WaveCoefficient(self, expr, args):
        """
        Handle the Hermitian conjugate of a WaveCoefficient.
        For a WaveCoefficient:
            amplitude -> complex conjugate of amplitude
            frequency -> same (assuming real frequency)
            phase -> negate the phase
            Other attributes are preserved, but transitions i and j are swapped.
        """
        amplitude = expr.amplitude.conj()
        frequency = expr.frequency
        phase = -expr.phase 
        ion_indx = expr.laser_indx 
        laser_indx = expr.ion_indx  
        i = expr.j  
        j = expr.i 
        mode_indx = expr.mode_indx 

        return ci.WaveCoefficient(
            amplitude=amplitude,
            frequency=frequency,
            phase=phase,
            ion_indx=ion_indx,
            laser_indx=laser_indx,
            i=i,
            j=j,
            mode_indx=mode_indx,
        )

    def map_ConstantCoefficient(self, expr, args):
        """
        Handle the Hermitian conjugate of a ConstantCoefficient.
        Constants are typically real and unchanged under HC.
        If they are complex, take their complex conjugate.
        """
        # modify if constants can be complex
        if isinstance(expr.value, complex):
            return ci.ConstantCoefficient(expr.value.conj())
        else:
            return deepcopy(expr)  

    def apply_HC(self, expr):
        """
        Utility method to apply Hermitian conjugation to an expression.
        """
        return self.walk(expr, mode='HC')
    
class CoefficientHC(ci.Post):
    """Class for applying Hermitian conjugation to coefficients."""

    def __init__(self):
        super().__init__(CoefficientHCRule())

########################################################################################

def simplify_complex_subtraction(
    wc1: WaveCoefficient, wc2: WaveCoefficient
) -> WaveCoefficient:
    """Simplifies the difference between two WaveCoefficient objects with the same frequency.

    Args:
        wc1 (WaveCoefficient): The minuend WaveCoefficient object.
        wc2 (WaveCoefficient): The subtrahend WaveCoefficient object.

    Returns:
        WaveCoefficient: A new WaveCoefficient representing the simplified difference (wc1 - wc2).

    Raises:
        ValueError: If wc1.frequency != wc2.frequency.
    """
    if wc1.frequency != wc2.frequency:
        raise ValueError("The frequencies of the two WaveCoefficients must be the same.")

    A1, phi1 = wc1.amplitude, wc1.phase
    A2, phi2 = wc2.amplitude, wc2.phase

    # Compute the real and imaginary parts of the complex amplitudes
    re1 = A1 * cos(phi1)
    im1 = A1 * sin(phi1)
    re2 = A2 * cos(phi2)
    im2 = A2 * sin(phi2)

    # Subtract the complex amplitudes
    re = re1 - re2
    im = im1 - im2

    # Compute the new amplitude and phase
    amplitude = sqrt(re**2 + im**2)
    phase = atan2(im, re)

    # Normalize phase to be between -pi and pi
    if phase > pi:
        phase -= 2 * pi
    elif phase < -pi:
        phase += 2 * pi

    # Create a new WaveCoefficient with the simplified amplitude and phase
    return WaveCoefficient(
        amplitude=amplitude,
        frequency=wc1.frequency,
        phase=phase,
        ion_indx=wc1.ion_indx,
        laser_indx=wc1.laser_indx,
        mode_indx=wc1.mode_indx,
        i=wc1.i,
        j=wc1.j,
    )


