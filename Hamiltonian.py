from adiabatic_elimination import adiabatic_elimination, simple_adiabatic_elimination
from ld_rwa_approximations import approximate
from analog_Hamiltonian_AST import QutipConversion


from quantumion.compilerv2 import Post

class Hamiltonian():

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
        self.tree = adiabatic_elimination(self, threshold)

    def apply_single_photon_adiabatic_elimination(self, threshold):
        self.tree = simple_adiabatic_elimination(self.tree, threshold)

    def apply_ld_and_rwa_approximations(self, n_cutoff, ld_cond_th = 1e-2, rwa_cutoff = 'inf'):
        self.tree = approximate(tree = self.tree, n_cutoff = n_cutoff, timescale = self.args['timescale'], ld_cond_th = ld_cond_th, rwa_cutoff = rwa_cutoff)

    def convert_to_qutip(self):

        compiler = Post(QutipConversion())
        _ = compiler(self.tree)

        return compiler.children[0].operands[-1]
    
