import numpy as np
import oqd_compiler_infrastructure as ci
from numbers import Number
from pydantic import BaseModel, Field, NonNegativeInt
from trical.light_matter.utilities import NonNegativeAngularMomentumNumber, AngularMomentumNumber
from typing import Dict, Tuple, Any, List, Callable, Optional, Union

# Chamber Language
class Chamber(ci.VisitableBaseModel):
    chain: "Chain"
    lasers: list["Laser"]
    B: float
    Bhat: list

class VibrationalMode(ci.VisitableBaseModel):
    eigenfreq: float
    eigenvect: List[float]
    # def modecutoff(self, cutoff_value: int):
    #     self.eigenvect = self.eigenvect[:cutoff_value]

class Level(ci.VisitableBaseModel):
    principal: Optional[NonNegativeInt] = None
    spin: Optional[NonNegativeAngularMomentumNumber] = None
    orbital: Optional[NonNegativeAngularMomentumNumber] = None
    nuclear: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital_nuclear: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital_nuclear_magnetization: Optional[AngularMomentumNumber] = None
    energy: float

class Transition(ci.VisitableBaseModel):
    level1: Level
    level2: Level
    einsteinA: float
    multipole: str 

class Ion(ci.VisitableBaseModel):
    mass: float
    charge: float
    levels: Dict[str, Level]
    transitions: Dict[Tuple[str, str], Transition] = {}

class Chain(ci.VisitableBaseModel):
    ions: List[Ion] = Field(default_factory=list)
    modes: List[VibrationalMode] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

class Laser(ci.VisitableBaseModel):
    wavelength: float = None
    intensity: float = None
    polarization: List[float] = None
    wavevector: List[float] = None
    detuning: float = 0.0
    phase: float =0.0
    def detune(self, Delta):
        self.detuning = Delta

    class Config:
        arbitrary_types_allowed = True

 

# class Laser(BaseModel):
#     wavelength: Optional[float] = None
#     k_hat: Optional[List[float]] = None
#     I: Union[Number, Callable[[float], float]]
#     I_0: float = None
#     eps_hat: List[float] = Field(default_factory=lambda: [0, 0, 1])
#     phi: float = 0.0
#     detuning: float = 0.0

#     class Config:
#         arbitrary_types_allowed = True
