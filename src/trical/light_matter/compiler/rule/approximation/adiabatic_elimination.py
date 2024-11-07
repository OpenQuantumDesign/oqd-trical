from typing import Dict, Tuple
import itertools
import numpy as np

from oqd_compiler_infrastructure import (
    TypeReflectBaseModel,
    Post,
    RewriteRule,
    FixedPoint,
)

from ....interface import (
    OperatorKron,
    WaveCoefficient,
    Zero,
    KetBra,
    OperatorScalarMul,
    OperatorMul,
    Identity,
    Displacement,
    Laser,
)

class Elimination(RewriteRule):
    """Rewrite rule for eliminating Hamiltonian tree terms according to the two-photon, adiabatic elimination approximation

    Args:
        rabi_freqs (Dict[Tuple[int,int,int,int],float]): Dictionary mapping (ion_indx, laser_indx, i, j) -> rabi; acquired from GetRabiFrequencyDetunings's traversal
        detunings (Dict[Tuple[int,int,int,int],float]): Dictionary mapping (ion_indx, laser_indx, i, j) -> detuning; acquired from GetRabiFrequencyDetunings's traversal
        threshold (float): Adiabatic threshold for ratio of Rabi_eff/ Delta detuning. Default: 1e-2

    Attributes:
        levels_to_eliminate (Dict): Dictionary mapping ion_indx (int) -> set(int), a set of levels to eliminate
        new_two_level_systems (Dict):
        old_two_level_system_info (Dict):
        transformations (Dict): Dictionary mapping ion_indx (int) -> Dictionary mapping new two level system pair (tuple) -> two level system pairs replaced by it (List[tuples])
    """

    def __init__(self, rabi_freqs, detunings, threshold=1e-2):
        super().__init__()
        self.rabis = rabi_freqs
        self.detunings = detunings
        self.threshold = threshold

        # Info that needs to be accessible for adiabatic 'substitution'
        self.levels_to_eliminate = {}  # ion_indx (int) -> levels_to_eliminate set(int)
        self.new_two_level_systems = (
            {}
        )  # lvl pair tuple key> eff rabi, D(alpha) tuple value
        self.old_two_level_system_info = {"lasers": dict(), "mot_terms": dict()}
        self.transformations = {}

        # Iterate over pairs of rabi frequencies
        ion_indices = sorted(list(set([indx for indx, _, _, _ in rabi_freqs])))
        laser_indices = sorted(list(set([indx for _, indx, _, _ in rabi_freqs])))

        for ion_indx in ion_indices:
            self.new_two_level_systems[ion_indx] = {}
            self.old_two_level_system_info["mot_terms"][ion_indx] = {}
            for laser_indx in laser_indices:

                # For a given ion and laser index selection, retrieve the compatible i,j indices
                ij_pairs = []
                for ion_indx_inner, laser_indx_inner, i, j in rabi_freqs:
                    if ion_indx == ion_indx_inner and laser_indx == laser_indx_inner:
                        if (i, j) not in ij_pairs:
                            ij_pairs.append(
                                (i, j),
                            )

                # Assuming this is only a three level system
                rabi1 = self.rabis[(ion_indx, laser_indx, *ij_pairs[0])]
                rabi2 = self.rabis[(ion_indx, laser_indx, *ij_pairs[1])]

                Delta = self.detunings[(ion_indx, laser_indx, *ij_pairs[1])]

                if not Delta:
                    continue

                rabi_eff = rabi1 * rabi2 / (2 * Delta)

                if rabi_eff / Delta < self.threshold:

                    # Track the level they have in common (to eliminate), as well as the ion it belongs to!
                    common_lvl = list(set(ij_pairs[0]).intersection(set(ij_pairs[1])))[
                        0
                    ]
                    if ion_indx in self.levels_to_eliminate:
                        self.levels_to_eliminate[ion_indx].add(common_lvl)
                    else:
                        self.levels_to_eliminate[ion_indx] = {common_lvl}

                    # Eliminating this level will result in an effective new 2-level systems

                    remaining_lvls = set(ij_pairs[0]).symmetric_difference(
                        set(ij_pairs[1])
                    )

                    unique_pairs = list(itertools.combinations(list(remaining_lvls), 2))
                    unique_pairs = [sorted(pair, reverse=True) for pair in unique_pairs]

                    for i, j in unique_pairs:

                        self.transformations[ion_indx] = {(i, j): ij_pairs}

                        if (i, j) in self.new_two_level_systems[ion_indx]:
                            # Only keep the larger of the two effective Rabi frequencies
                            # There are two pairs of possible effective Rabi frequencies because have 2 transitions and 2 lasers

                            if (
                                rabi_eff
                                > self.new_two_level_systems[ion_indx][(i, j)][
                                    "rabi_eff"
                                ]
                            ):
                                self.new_two_level_systems[ion_indx][(i, j)][
                                    "rabi_eff"
                                ] = rabi_eff
                                self.new_two_level_systems[ion_indx][(i, j)][
                                    "lasers"
                                ].append(laser_indx)
                            else:
                                continue
                        else:
                            self.new_two_level_systems[ion_indx][(i, j)] = {
                                "rabi_eff": rabi_eff,
                                "lasers": [laser_indx],
                            }

    def map_OperatorMul(self, model):
        """Method for killing Hamiltonian tree terms that are in self.levels_to_eliminate for a particular ion_indx

        Args:
            model (OperatorMul): looks for instances of OperatorMul with OperatorScalarMul, OperatorKron children

        Returns:
            model (OperatorMul): if |iXj| coupling is NOT to be eliminated in ion with index ion_indx
            (Zero): if |iXj| coupling SHOULD be eliminated in ion with index ion_indx
        """

        op1 = model.op1
        op2 = model.op2

        if isinstance(op1, OperatorScalarMul) and isinstance(op2, OperatorKron):

            phase = op1.coeff.phase

            mot_term = op2

            ic = op1.coeff

            i, j = ic.i, ic.j
            ion_indx, laser_indx = ic.ion_indx, ic.laser_indx
            if ion_indx in self.levels_to_eliminate and (
                i in self.levels_to_eliminate[ion_indx]
                or j in self.levels_to_eliminate[ion_indx]
            ):

                self.old_two_level_system_info["lasers"][laser_indx] = {"phase": phase}

                mot_terms_n = self.old_two_level_system_info["mot_terms"][ion_indx]

                if laser_indx in mot_terms_n:
                    mot_terms_n[laser_indx][(i, j)] = mot_term

                else:
                    mot_terms_n[laser_indx] = {(i, j): mot_term}

                return Zero()

        else:
            return model

def adiabatic_elimination(H, threshold):
    """Master function for calling ReWrite rules and helper functions for performing two photon, adiabatic elimination

    Args:
        H (OperatorAdd): Hamiltonian tree to approximate, with terms united using OperatorAdd
        threshold (float): Adiabatic threshold on ratio Rabi_eff/Delta detuning

    Returns:
        final_tree (OperatorAdd): Hamiltonian tree post approximation step
    """

    tree = H.tree
    N, L, ion_N_levels, mode_cutoffs = H.N, H.L, H.ion_N_levels, H.mode_cutoffs

    get_freqs = GetRabiFrequenciesDetunings()

    _ = Post(get_freqs)(tree)

    rabis = get_freqs.rabis
    detunings = get_freqs.detunings

    eliminate = Elimination(rabi_freqs=rabis, detunings=detunings, threshold=threshold)
    eliminator = Post(eliminate)
    chopped_tree = eliminator(tree)

    final_tree = substitute(
        chopped_tree=chopped_tree,
        old_info=eliminate.old_two_level_system_info,
        two_level_info=eliminate.new_two_level_systems,
        N=N,
        L=L,
        ion_N_levels=ion_N_levels,
        mode_cutoffs=mode_cutoffs,
        detunings=detunings,
        transformations=eliminate.transformations,
    )

    simplifier = FixedPoint(Post(SimplifyZero()))

    final_tree = simplifier(final_tree)

    return final_tree
