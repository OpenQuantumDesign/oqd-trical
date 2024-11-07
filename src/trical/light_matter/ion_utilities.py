import numpy as np
from qutip import qeye, tensor, basis
from scipy.special import genlaguerre as L
from trical.backend.qutip.passes import QuantumOperator
from trical.light_matter.interface.Chamber import VibrationalMode, Level, Transition
from trical.misc.polynomial import PolynomialPotential
from trical.misc.multispecies import TrappedIons

# ION UTILITIES

def get_level(levels, name):
    """
    Accesses a level by its alias or label.

    Args:
        levels (dict): Dictionary containing level information.
        name (str): Alias or label of the level.

    Returns:
        dict: The Level object associated with the user-defined name.

    Raises:
        KeyError: If the level name is not found.
    """
    if name not in levels:
        raise KeyError(f"Level {name} not found.")
    return levels[name]

def get_transition(transitions, transition_key):
    """
    Accesses a transition using the transition key.

    Args:
        transitions (dict): Dictionary of transition objects.
        transition_key (tuple): Tuple of the form (alias1, alias2) or (manifold1, manifold2).

    Returns:
        dict: The Transition object associated with the alias-alias or manifold-manifold pair.

    Raises:
        ValueError: If the transition key doesn't exist.
    """
    if transition_key in transitions:
        return transitions[transition_key]
    elif (transition_key[1], transition_key[0]) in transitions:
        return transitions[(transition_key[1], transition_key[0])]
    else:
        raise ValueError("Transition key not found!")

from qutip import basis

def select_levels(levels, full_transitions, selected_levels):
    """
    Selects and overrides levels with a specified subset. Constructs possible transitions between the selected levels.

    Args:
        levels (dict): Dictionary of available levels with level details.
        full_transitions (dict): Dictionary of all possible transitions between levels.
        selected_levels (list): List of tuples specifying (manifold, m_F, alias) for selected levels.

    Returns:
        tuple: (updated levels dictionary, alias-to-manifold mapping, state dictionary, transition subset dictionary).
    """
    if not selected_levels:
        return levels, {}, {}, {}

    alias_to_level = {}
    alias_to_manifold = {}
    state = {}
    N_levels = len(selected_levels)

    for l, (manifold, m_F, alias) in enumerate(selected_levels):
        full_lvl = levels[manifold]

        level = {
            'principal': full_lvl.principal,
            'orbital': full_lvl.orbital,
            'spin_orbital': full_lvl.spin_orbital,
            'nuclear': full_lvl.nuclear,
            'spin_orbital_nuclear': full_lvl.spin_orbital_nuclear,
            'energy': full_lvl.energy,
            'spin_orbital_nuclear_magnetization': m_F
        }

        alias_to_manifold[alias] = manifold
        alias_to_level[alias] = level

        state[alias] = basis(N_levels, N_levels - 1 - l)

    # Construct possible transitions between the levels
    alias_pairs = []
    alias_pairs = [
        (alias1, alias2) for alias1 in alias_to_level.keys()
        for alias2 in alias_to_level.keys()
        if alias1 != alias2
    ]

    transition_subset = {}
    for alias1, alias2 in alias_pairs:
        manifold1, manifold2 = alias_to_manifold[alias1], alias_to_manifold[alias2]
        full_transition_key = (manifold1, manifold2)

        if full_transition_key not in full_transitions.keys():
            if (manifold2, manifold1) in full_transitions.keys():
                full_transition_key = (manifold2, manifold1)
            else:
                continue

        multipole = full_transitions[full_transition_key].multipole
        einsteinA = full_transitions[full_transition_key].einsteinA

        # Level 1 should always be the one lower in energy
        if alias_to_level[alias1]['energy'] > alias_to_level[alias2]['energy']:
            level1 = alias_to_level[alias2]
            level2 = alias_to_level[alias1]
        else:
            level1 = alias_to_level[alias1]
            level2 = alias_to_level[alias2]

        transition = {
            'level1': level1,
            'level2': level2,
            'einsteinA': einsteinA,
            'multipole': multipole,
        }

        transition_subset[(alias1, alias2)] = transition

    return alias_to_level, alias_to_manifold, state, transition_subset


def ion_projector(ions, selected_modes, ion_numbers, names):
    """Full Hilbert space projector onto internal state.

    Args:
        ions (List[Ion]): List of Ion objects.
        selected_modes (List[VibrationalMode]): List of selected vibrational modes.
        ion_numbers (Union[int, List[int]]): Index or list of indices of ions.
        names (Union[str, List[str]]): Names of the states.

    Returns:
        QuantumOperator: The projector in the full Hilbert space.
    """
    mot_buffer = [qeye(len(selected_modes)) for _ in selected_modes]

    if isinstance(names, str) and isinstance(ion_numbers, int):
        ion_buffer = [qeye(len(ion.levels)) for ion in ions]
        state_index = list(ions[ion_numbers - 1].levels.keys()).index(names)
        ket = basis(len(ions[ion_numbers - 1].levels), state_index)
        ion_buffer[ion_numbers - 1] = ket * ket.dag()
        name = names
    elif isinstance(names, list):
        ket = tensor(
            *[
                basis(len(ions[ion_numbers[j] - 1].levels), list(ions[ion_numbers[j] - 1].levels.keys()).index(names[j]))
                for j in range(len(names))
            ]
        )
        ion_buffer = []
        for j in range(len(ions)):
            if j + 1 not in ion_numbers:
                ion_buffer.append(qeye(len(ions[j].levels)))
            elif j + 1 == ion_numbers[0]:
                ion_buffer.append(ket * ket.dag())
        name = "".join(names)
    else:
        raise ValueError("Invalid types for ion_numbers or names")

    return tensor(*ion_buffer, *mot_buffer)

def initialize_trapped_ions(ions, trap_freqs):
    """Initialize the trapped ions and calculate equilibrium positions and vibrational modes.

    Args:
        ions (List[Ion]): List of Ion objects.
        trap_freqs (List[float]): List of trap frequencies in Hz [omega_x, omega_y, omega_z].

    Returns:
        Tuple: A tuple containing (modes, equilibrium_positions)
    """
    N = len(ions)
    mass = ions[0].mass

    omega_x = 2 * np.pi * trap_freqs[0]
    omega_y = 2 * np.pi * trap_freqs[1]
    omega_z = 2 * np.pi * trap_freqs[2]

    alpha = np.zeros((3, 3, 3))
    alpha[2, 0, 0] = mass * (omega_x) ** 2 / 2
    alpha[0, 2, 0] = mass * (omega_y) ** 2 / 2
    alpha[0, 0, 2] = mass * (omega_z) ** 2 / 2

    # Create the polynomial potential and trapped ion system
    pp = PolynomialPotential(alpha, N=N)
    ti = TrappedIons(N, pp, m=mass)
    ti.principle_axis()

    eigenfreqs = ti.w_pa
    eigenvects = ti.b_pa

    modes = []
    for l in range(len(eigenfreqs)):
        if 0 <= l <= N - 1:
            axis = np.array([1, 0, 0])
        elif N <= l <= 2 * N - 1:
            axis = np.array([0, 1, 0])
        elif 2 * N <= l <= 3 * N - 1:
            axis = np.array([0, 0, 1])
        else:
            raise ValueError("Frequency direction sorting went wrong :(")

        modes.append(
            VibrationalMode(
                eigenfreq=eigenfreqs[l],
                eigenvect=eigenvects[l],
                axis=axis.tolist(),
                N=10
            )
        )

    equilibrium_positions = ti.equilibrium_position()
    return modes, equilibrium_positions



