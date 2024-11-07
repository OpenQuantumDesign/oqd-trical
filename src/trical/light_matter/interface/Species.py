#from .ion import *
from trical.light_matter.interface.Chamber import Ion, Level, Transition
from trical.light_matter.ion_utilities import select_levels
from qutip import Qobj
from typing import Dict, Any
from pydantic import Field

########################################################################################
#fine splitting and hyperfine splitting needs to be added

levels = {
    "S1/2": Level(
        principal=4,
        orbital=0,
        spin_orbital=1 / 2,
        spin_orbital_nuclear=1 / 2,
        energy=0,
        nuclear=0,
    ),
    "D3/2": Level(
        principal=3,
        orbital=2,
        spin_orbital=3 / 2,
        spin_orbital_nuclear=3 / 2,
        energy=4.09335071228e14,
        nuclear=0,
    ),
    "D5/2": Level(
        principal=3,
        orbital=2,
        spin_orbital=5 / 2,
        spin_orbital_nuclear=5 / 2,
        energy=4.1115503183857306e14,
        nuclear=0,
    ),
    "P1/2": Level(
        principal=4,
        orbital=1,
        spin_orbital=1 / 2,
        spin_orbital_nuclear=1 / 2,
        energy=7.554e14,
        nuclear=0,
    ),
    "P3/2": Level(
        principal=4,
        orbital=1,
        spin_orbital=3 / 2,
        spin_orbital_nuclear=3 / 2,
        energy=7.621e14,
        nuclear=0,
    ),
}

transitions = {
    ("S1/2", "D5/2"): Transition(
        level1=levels["S1/2"], level2=levels["D5/2"], einsteinA=8.562e-1, multipole="E2"
    ),
    ("S1/2", "P1/2"): Transition(
        level1=levels["S1/2"], level2=levels["P1/2"], einsteinA=1.299e8, multipole="E1"
    ),
    ("D3/2", "P1/2"): Transition(
        level1=levels["D3/2"], level2=levels["P1/2"], einsteinA=1.060e7, multipole="E1"
    ),
    ("S1/2", "D3/2"): Transition(
        level1=levels["S1/2"], level2=levels["D3/2"], einsteinA=9.259e-1, multipole="E2"
    ),
    ("S1/2", "P3/2"): Transition(
        level1=levels["S1/2"], level2=levels["P3/2"], einsteinA=1.351e8, multipole="E1"
    ),
    ("D3/2", "P3/2"): Transition(
        level1=levels["D3/2"], level2=levels["P3/2"], einsteinA=1.110e6, multipole="E1"
    ),
    ("D5/2", "P3/2"): Transition(
        level1=levels["D5/2"], level2=levels["P3/2"], einsteinA=9.901e6, multipole="E1"
    ),
}

class Ca40(Ion):
    """Class representing calcium 40 species

    Args:
        selected_levels (list): List of tuples of the form (manifold, alias, m_F); restrict ion's internal Hilbert space to these
    """
    def __init__(self):
        super().__init__(
            levels=levels,       
            transitions=transitions,
            mass=6.635943757345042e-26,
            charge=1,
        )

    class Config:
        arbitrary_types_allowed = True
