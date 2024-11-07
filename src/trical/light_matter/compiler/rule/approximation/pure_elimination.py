import numpy as np
from typing import Dict, Tuple
from oqd_compiler_infrastructure import (
    Post,
    RewriteRule,
    FixedPoint,
)

from ....interface import (
    OperatorKron,
    Zero,
    OperatorScalarMul,
)

class PureElimination(RewriteRule):
    """Rewrite rule for eliminating weakly coupled terms in the single-photon adibatic elimination approximation step

    Args:
        rabi_freqs (Dict): Dictionary mapping (ion_indx, laser_indx, i, j) -> rabi; acquired from GetRabiFrequencyDetunings's traversal
        detunings (Dict): Dictionary mapping (ion_indx, laser_indx, i, j) -> detuning; acquired from GetRabiFrequencyDetunings's traversal
        threshold (float): Adiabatic threshold on transition probability between 2 levels. Default: 1e-2

    Attributes:
        transition_probs (Dict): Dictionary mapping (ion_indx, laser_indx, i, j) -> transition probability P
        weakly_coupled_transitions (List): List of (ion_indx, i,j) tuples of weak couplings on |iXj| for ion_indx

    """

    def __init__(self, rabi_freqs, detunings, threshold=1e-2):
        super().__init__()
        self.rabis = rabi_freqs
        self.detunings = detunings
        self.threshold = threshold

        self.transition_probs = {}
        # List of (n,i,j) triplets corresponding to two weakly coupled levels in ion n
        self.weakly_coupled_transtions = []

        for tup_key in self.rabis:
            ion_indx, _, i, j = tup_key

            rabi = self.rabis[tup_key]
            Delta = self.detunings[tup_key]

            # Transition probability: valid for large detunings
            if not Delta:
                continue

            P = rabi**2 / Delta**2
            self.transition_probs[tup_key] = P
            if P < self.threshold:
                self.weakly_coupled_transtions.append(
                    (ion_indx, i, j),
                )

        print("WEAKLY COUPLED TRANSITIONS:", self.weakly_coupled_transtions)
        print("TRANSITION PROBS:", self.transition_probs)

    def map_OperatorMul(self, model):
        """
        Args:
            model (OperatorMul): OperatorMul object; targeting those with OperatorScalarMul and OperatorKron children

        Returns:
            Zero() (Zero): |iXj| in ion ion_indx is in self.weakly_coupled_transitions
            model (OperatorMul): otherwise
        """

        op1 = model.op1
        op2 = model.op2

        if isinstance(op1, OperatorScalarMul) and isinstance(op2, OperatorKron):

            ic = op1.coeff  # "indexed" coefficient

            i, j = ic.i, ic.j
            ion_indx, _ = ic.ion_indx, ic.laser_indx

            if (ion_indx, i, j) in self.weakly_coupled_transtions:
                return Zero()


def simple_adiabatic_elimination(tree, threshold):
    """Master function for calling ReWrite rules and helper functions for performing single photon adiabatic elimination

    Args:
        tree (OperatorAdd): Hamiltonian tree to approximate, with terms united using OperatorAdd
        threshold (float): Adiabatic threshold on ratio transition probability between two levels

    Returns:
        simplifier(chopped_tree) (OperatorAdd): Hamiltonian tree post approximation step
    """

    get_freqs = GetRabiFrequenciesDetunings()

    _ = Post(get_freqs)(tree)

    rabis = get_freqs.rabis
    detunings = get_freqs.detunings

    eliminate = PureElimination(
        rabi_freqs=rabis, detunings=detunings, threshold=threshold
    )
    eliminator = Post(eliminate)
    chopped_tree = eliminator(tree)

    simplifier = FixedPoint(Post(SimplifyZero()))

    return simplifier(chopped_tree)