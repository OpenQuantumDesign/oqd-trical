from midstack.interface.base import TypeReflectBaseModel

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
        try:
            return CoefficientMul(expr1=self, expr2=other)
        except:
            return other * self

    pass


########################################################################################


class ConstantCoefficient(Coefficient):
    """Class for coefficients that do not change in time"""

    value: float


class WaveCoefficient(Coefficient):
    """Class for coefficients of the form $A e^{i (\omega t + \phi)}$

    Attributes:
        amplitude (float): ampitude of the wave $A$
        frequency (float): frequency of the wave $\omega$
        phase (float): phase of the wave $\phi$
        ion_indx (int): identification index of the ion as defined when instantiating the ion
        laser_indx (int): identification index of the laser as defined when instantiating the laser
        mode_indx (int): identification index of the mode as defined when instantiating the mode
        i (int): $i$ in $|i \\rangle \langle j|$ if WaveCoefficient is a prefactor for a KetBra
        j (int): $j$ in $|i \\rangle \langle j|$ if WaveCoefficient is a prefactor for a KetBra
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
