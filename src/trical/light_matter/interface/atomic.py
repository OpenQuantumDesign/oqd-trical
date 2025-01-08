from __future__ import annotations

from typing import List, Union, Optional, Annotated, Literal

from pydantic import conlist, NonNegativeInt, NonNegativeFloat, AfterValidator

from oqd_compiler_infrastructure import TypeReflectBaseModel
from oqd_core.interface.math import CastMathExpr

########################################################################################


class AtomicCircuit(TypeReflectBaseModel):
    """
    Class representing a trapped-ion experiment in terms of light-matter interactons.

    Attributes:
        system: The trapped-ion system.
        protocol: Pulse program for the trapped-ion experiment referenced to the trapped-ion system.

    """

    system: System
    protocol: Protocol


########################################################################################


class Beam(TypeReflectBaseModel):
    """
    Class representing a referenced optical channel/beam for the trapped-ion device.

    Attributes:
        transition: Transition that the optical channel/beam is referenced to.
        rabi: Rabi frequency of the referenced transition driven by the beam.
        detuning: Detuning away from the referenced transition.
        phase: Phase relative to the ion's clock.
        polarization: Polarization of the beam.
        wavevector: Wavevector of the beam.
        target: Index of the target ion of the beam.
    """

    transition: Transition
    rabi: CastMathExpr
    detuning: CastMathExpr
    phase: CastMathExpr
    polarization: conlist(float, max_length=3, min_length=3)
    wavevector: conlist(float, max_length=3, min_length=3)
    target: int


class Pulse(TypeReflectBaseModel):
    """
    Class representing the application of the beam for some duration.

    Attributes:
        beam: Optical channel/beam to turn on.
        duration: Period of time to turn the optical channel on for.

    """

    beam: Beam
    duration: float


class Protocol(TypeReflectBaseModel):
    """
    Class representing a light-matter interaction protocol/pulse program for the optical channels/beams.
    """

    pass


class ParallelProtocol(Protocol):
    """
    Class representing the parallel composition of a list of pulses or subprotocols.

    Attributes:
        sequence: List of pulses or subprotocols to compose together in a parallel fashion.
    """

    sequence: List[Union[Pulse, Protocol]]


class SequentialProtocol(Protocol):
    """
    Class representing the sequential composition of a list of pulses or subprotocols.

    Attributes:
        sequence: List of pulses or subprotocols to compose together in a sequntial fashion.
    """

    sequence: List[Union[Pulse, Protocol]]


########################################################################################


def is_halfint(v: float) -> bool:
    """
    Function that verifies a number is an integer or half-integer.

    Args:
        v: Number to verify.
    """
    if not (v * 2).is_integer():
        raise ValueError()
    return v


########################################################################################


AngularMomentumNumber = Annotated[float, AfterValidator(is_halfint)]
"""
A valid positive or negative integer or half-integer for angular momentum.
"""
NonNegativeAngularMomentumNumber = Annotated[
    NonNegativeFloat, AfterValidator(is_halfint)
]
"""
A valid non-negative integer or half-integer for angular momentum.
"""

# ########################################################################################


class Level(TypeReflectBaseModel):
    """ "
    Class representing an electronic energy level of an ion.

    Attributes:
        principal: Principal quantum number.
        spin: Spin of an electron.
        orbital: Orbital angular momentum of an electron.
        nuclear: Nuclear angular momentum.
        spin_orbital: Angular momentum of the spin-orbital coupling.
        spin_orbital_nuclear: Angular momentum of the spin-orbital-nuclear coupling.
        spin_orbital_nuclear_magnetization: Magnetization of the spin-orbital-nuclear coupled angular momentum.
        energy: Energy of the electronic state.

    """

    principal: Optional[NonNegativeInt] = None
    spin: Optional[NonNegativeAngularMomentumNumber] = None
    orbital: Optional[NonNegativeAngularMomentumNumber] = None
    nuclear: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital_nuclear: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital_nuclear_magnetization: Optional[AngularMomentumNumber] = None
    energy: float

    # @model_validator(mode="after")
    # def orbital_validate(self):
    #     if self.orbital >= self.principal:
    #         raise ValueError("Invalid orbital quantum # (L)")
    #     return self

    # @model_validator(mode="after")
    # def spin_orbital_validate(self):
    #     if (
    #         self.spin_orbital < abs(self.spin - self.orbital)
    #         or self.spin_orbital > self.spin + self.orbital
    #     ):
    #         raise ValueError("Invalid spin orbital quantum # (J)")
    #     return self

    # @model_validator(mode="after")
    # def spin_orbital_nuclear_validate(self):
    #     if (
    #         self.spin_orbital_nuclear < abs(self.spin_orbital - self.nuclear)
    #         or self.spin_orbital_nuclear > self.spin_orbital + self.nuclear
    #     ):
    #         raise ValueError("Invalid spin orbital nuclear quantum # (F)")
    #     return self

    # @model_validator(mode="after")
    # def spin_orbital_nuclear_magnetization_validate(self):
    #     if abs(self.spin_orbital_nuclear_magnetization) > self.spin_orbital_nuclear:
    #         raise ValueError("Invalid spin orbital nuclear magnetization (m_F)")
    #     elif not (
    #         self.spin_orbital_nuclear_magnetization - self.spin_orbital_nuclear
    #     ).is_integer():
    #         raise ValueError("Invalid spin orbital nuclear magnetization (m_F)")
    #     return self


class Transition(TypeReflectBaseModel):
    """
    Class representing a transition between electronic states of an ion.

    Attributes:
        level1: Energy level 1.
        level2: Energy level 2.
        einsteinA: Einstein A coefficient that characterizes the strength of coupling between energy level 1 and 2.

    """

    level1: Level
    level2: Level
    einsteinA: float
    multipole: Literal["E1", "E2", "M1"]


class Ion(TypeReflectBaseModel):
    """
    Class representing an ion.

    Attributes:
        mass: Mass of the ion.
        charge: Charge of the ion.
        levels: Electronic energy levels of the ion.
        transitions: Allowed transitions in the ion.
        position: Spatial position of the ion.
    """

    mass: float
    charge: float
    levels: List[Level]
    transitions: List[Transition]
    position: List[float]


########################################################################################


class Phonon(TypeReflectBaseModel):
    """
    Class representing a collective phonon mode of the trapped-ion system.

    Attributes:
        energy: Quanta of energy for the phonon mode.
        eigenvector: Profile of the phonon mode in terms of the vibration of the ions.

    """

    energy: float
    eigenvector: List[float]


########################################################################################


class System(TypeReflectBaseModel):
    """
    Class representing a trapped-ion system.

    Attributes:
        ions: List of ions in the trapped-ion system.
        modes: List of collective phonon modes for the trapped-ion system.

    """

    ions: List[Ion]
    modes: List[Phonon]
