from typing import Dict, Tuple, Optional
from typing_extensions import Annotated

from pydantic import (
    NonNegativeInt,
    AfterValidator,
    NonNegativeFloat,
)

from qutip import basis

from midstack.interface.base import TypeReflectBaseModel

########################################################################################


def is_halfint(v: float) -> float:
    if not (v * 2).is_integer():
        raise ValueError()
    return v


AngularMomentumNumber = Annotated[float, AfterValidator(is_halfint)]
NonNegativeAngularMomentumNumber = Annotated[
    NonNegativeFloat, AfterValidator(is_halfint)
]


class Level(TypeReflectBaseModel):
    """Class for representing level in ion's Hilbert space

    Attributes:
        principal (NonNegativeInt): quantum number N
        spin (NonNegativeAngularMomentumNumber): quantum number S (spin angular momentum)
        orbital (NonNegativeAngularMomentumNumber): quantum number L (orbital angular momentum)
        nuclear (NonNegativeAngularMomentumNumber): quantum number I (nuclear angular momenutm)
        spin_orbital (NonNegativeAngularMomentumNumber): quantum number J = L + S
        spin_orbital_nuclear (NonNegativeAngularMomentumNumber): quantum F = J + I
        spin_orbital_nuclear_magnetization (NonNegativeAngularMomentumNumber): M_F
        energy (float): energy in units of hbar (frequency) relative to the ground state
    """

    principal: Optional[NonNegativeInt] = None
    spin: Optional[NonNegativeAngularMomentumNumber] = None
    orbital: Optional[NonNegativeAngularMomentumNumber] = None
    nuclear: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital_nuclear: Optional[NonNegativeAngularMomentumNumber] = None
    spin_orbital_nuclear_magnetization: Optional[AngularMomentumNumber] = None
    energy: float


class Transition(TypeReflectBaseModel):
    """Class for representing transition between 2 levels in an ion's Hilbert space

    Attributes:
        level1 (Level):
        level2 (Level):
        einsteinA (float): Einstein A coefficient for transition
        multipole (str): 'E1' for dipole or 'E2' for quadrupole allowed transitions
    """

    level1: Level
    level2: Level
    einsteinA: float
    multipole: str


class Ion(TypeReflectBaseModel):
    """Class for representing an ion

    Attributes:
        mass (float): ion's mass in kg
        charge (float): ion's charge in units of e
        levels (Dict): Dictionary mapping string alias/name to Transition Level object
        full_transitions (Dict): Dictionary mapping manifold-manifold (str-str) tuples to Transition object
        transitions (Dict): Dictionary mapping sublevel alias - sublevel alias (str-str) to Transition object
        alias_to_manifold (Dict): Dictionary mapping alias (e.g. 'S') to manifold name (e.g. 'S1/2')
        N_levels (int): number of levels in the ion's Hilbert space
        state (Dict): Dictionary mapping alias to Level object

    """

    mass: float
    charge: float
    levels: Dict[str, Level]  # used to be List[Level]
    full_transitions: Dict[Tuple[str, str], Transition]  # used to be List[Transition]
    transitions: Dict[Tuple[str, str], Transition] = {}
    # alias_to_level: Dict[str, str] = {}
    alias_to_manifold: Dict[str, str] = {}

    N_levels: int = 0
    state: Dict[str, None] = {}

    def select_levels(self, selected_levels):
        # Overrides all levels with specified subset if entered by user. Otherwise, work with whole space

        if selected_levels:
            alias_to_level = {}

            # We'll want to construct Level objects based on the info provided by the user
            self.N_levels = len(selected_levels)
            for l, (manifold, m_F, alias) in enumerate(selected_levels):
                # e.g. 'S1/2', 0.5, 'S+'

                full_lvl = self.levels[manifold]

                level = Level(
                    principal=full_lvl.principal,
                    orbital=full_lvl.orbital,
                    spin_orbital=full_lvl.spin_orbital,
                    nuclear=full_lvl.nuclear,
                    spin_orbital_nuclear=full_lvl.spin_orbital_nuclear,
                    energy=full_lvl.energy,
                    spin_orbital_nuclear_magnetization=m_F,
                )

                self.alias_to_manifold[alias] = manifold
                alias_to_level[alias] = level

                self.state[alias] = basis(
                    self.N_levels, self.N_levels - 1 - l
                )  # ranges from 0 to self.N_levels-1

            # Construct possible transitions between the levels
            self.levels = alias_to_level
            # First let's group the aliases into unique pairs
            alias_pairs = []

            for alias1 in self.levels.keys():
                for alias2 in self.levels.keys():
                    if (
                        alias1 == alias2
                        or (alias1, alias2) in alias_pairs
                        or (alias2, alias1) in alias_pairs
                    ):
                        continue
                    alias_pairs.append(
                        (alias1, alias2),
                    )

            # What transitions are allowed?

            transition_subset = {}
            for alias1, alias2 in alias_pairs:

                manifold1, manifold2 = (
                    self.alias_to_manifold[alias1],
                    self.alias_to_manifold[alias2],
                )

                full_transition_key = (manifold1, manifold2)

                if full_transition_key not in self.full_transitions.keys():
                    if (manifold2, manifold1) in self.full_transitions.keys():
                        full_transition_key = (manifold2, manifold1)
                    else:
                        continue

                multipole = self.full_transitions[full_transition_key].multipole
                einsteinA = self.full_transitions[full_transition_key].einsteinA

                # Level 1 should always be the one lower in energy
                if self.levels[alias1].energy > self.levels[alias2].energy:
                    level1 = self.levels[alias2]
                    level2 = self.levels[alias1]
                else:
                    level1 = self.levels[alias1]
                    level2 = self.levels[alias2]

                transition = Transition(
                    level1=level1,
                    level2=level2,
                    einsteinA=einsteinA,
                    multipole=multipole,
                )

                transition_subset[(alias1, alias2)] = transition

            # Replace all transitions in the Hilbert space with just the allowed ones from the
            # level subset specified by the user
            self.transitions = transition_subset

    def level(self, name):
        """Wrapper function for accessing level

        Args:
            name (str): alias or label e.g. 's' or 'S1/2, respectively

        Returns:
            (Level): Level object associated with user-defined name
        """

        return self.levels[name]

    def transition(self, transition_key):
        """Wrapper function for accessing transition

        Args:
            transition_key (tuple): tuple of the form (alias1, alias2) or (manifold1, manifold2), where constituents are strings

        Returns:
            (Transition): Transition object associated with alias-alias or manifold-manifold pair

        Raises:
            ValueError: if the transition key doesn't exist
        """

        if transition_key in self.transitions.keys():
            return self.transitions[transition_key]
        elif (transition_key[1], transition_key[0]) in self.transitions.keys():
            return self.transitions[(transition_key[1], transition_key[0])]

        else:
            raise ValueError("Transition key not found!")
