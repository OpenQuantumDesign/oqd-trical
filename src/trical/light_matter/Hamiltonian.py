from trical.light_matter.adiabatic_elimination import (
    adiabatic_elimination,
    simple_adiabatic_elimination,
)
from trical.light_matter.ld_rwa_approximations import approximate
from trical.light_matter.analog_Hamiltonian_AST import QutipConversion


from oqd_compiler_infrastructure import Post


class Hamiltonian:
    """Class representing the Hamiltonian

    Args:
        tree (OperatorAdd): Hamiltonian tree whose terms are joined via OperatorAdd's
        args (Dict): Dictionary of the form {'timescale': timescale, 'chamber': chamber}
        N (int): number of ions in the system
        M (int): number of lasers in the system
        L (int): number of motional modes under consideration
        ion_N_levels (Dict): Dictionary mapping ion indices to number of levels in its Hilbert space
        mode_cutoffs (Dict): Dictionary mapping mode indices to the phonon cutoff
    """

    def __init__(self, tree, args, N, M, L, ion_N_levels, mode_cutoffs):
        self.tree = tree
        self.args = args
        # Total number of ions, lasers, and motional modes, respectively
        self.N = N
        self.M = M
        self.L = L
        self.ion_N_levels = ion_N_levels
        self.mode_cutoffs = mode_cutoffs

    def apply_adiabatic_elimination(self, threshold):
        """Convenience method for applying two photon adiabatic elimination"""
        self.tree = adiabatic_elimination(self, threshold)

    def apply_single_photon_adiabatic_elimination(self, threshold):
        """Convenience method for applying single photon adiabatic elimination"""
        self.tree = simple_adiabatic_elimination(self.tree, threshold)

    def apply_ld_and_rwa_approximations(
        self, n_cutoff, ld_cond_th=1e-2, rwa_cutoff="inf"
    ):
        """Convenience method for applying the Lamb-Dicke and rotating wave approximations"""
        self.tree = approximate(
            tree=self.tree,
            n_cutoff=n_cutoff,
            timescale=self.args["timescale"],
            ld_cond_th=ld_cond_th,
            rwa_cutoff=rwa_cutoff,
        )

    def convert_to_qutip(self):
        """Convenience method for converting a Hamiltonian tree to a Qutip QobjEvo"""

        compiler = Post(QutipConversion())
        _ = compiler(self.tree)

        return compiler.children[0].operands[-1]
