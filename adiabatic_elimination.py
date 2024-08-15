from analog_Hamiltonian_AST import OperatorKron, WaveCoefficient, Zero, KetBra, OperatorScalarMul, OperatorMul, Identity, Displacement
import itertools
from structures import Laser
import numpy as np
from quantumion.interface.base import TypeReflectBaseModel
from quantumion.compilerv2 import Post, RewriteRule, FixedPoint

def extract_indices_dims(ekron):
    """
    Recursive function that takes in an 'extended kron' argument, which is an arbitrary sequences of kron operators
    and returns the indices of the nested KetBra.

    """

    op1 = ekron.op1
    op2 = ekron.op2

    if isinstance(op1, KetBra):
        return op1.i, op1.j, op1.dims
    
    elif isinstance(op2, KetBra):
        return op2.i, op2.j, op2.dims
    
    if isinstance(op1, OperatorKron):
        return extract_indices_dims(op1) 
    
    elif isinstance(op2, OperatorKron):
        return extract_indices_dims(op2)



def extract_alphas(ekron):

    alphas = []
    op1 = ekron.op1
    op2 = ekron.op2
    
    while True:

        if isinstance(op1, OperatorKron) and not isinstance(op2, OperatorKron):
            if isinstance(op2, Displacement):
                alphas.append(op2.alpha)
            
            # Step left
            op1 = op1.op1
            op2 = op1.op2
            continue

        if isinstance(op2, OperatorKron) and not isinstance(op1, OperatorKron):
            
            if isinstance(op1, Displacement):
                alphas.append(op1.alpha)
            
            # Step right
            op1 = op2.op1
            op2 = op2.op2
            continue

        # If neither are OperatorKron's, we've made it to the end of the tree
        if not isinstance(op1, OperatorKron) and not isinstance(op2, OperatorKron):

            # Check tree leaves

            if isinstance(op1, Displacement):
                alphas.append(op1.alpha)
            if isinstance(op2, Displacement):
                alphas.append(op2.alpha)
            break

    return alphas





class GetRabiFrequenciesDetunings(RewriteRule):

    def __init__(self):
        super().__init__()

        self.rabis = {}
        self.detunings = {}


    def map_OperatorScalarMul(self, model):
        coeff = model.coeff
        op = model.op

        if isinstance(op, OperatorKron) and isinstance(coeff, WaveCoefficient):
            rabi = 2 * coeff.amplitude

            # The signed detuning info is stored in the frequency argument
            Delta = coeff.frequency

            # Index info
            i,j = coeff.i, coeff.j
            ion_indx, laser_indx = coeff.ion_indx, coeff.laser_indx

            
            self.rabis[(ion_indx, laser_indx, i, j)] = rabi
            self.detunings[(ion_indx, laser_indx, i, j)] = Delta
            
        return model


class Elimination(RewriteRule):

    def __init__(self, rabi_freqs, detunings, threshold = 1e-2):
        super().__init__()
        self.rabis = rabi_freqs
        self.detunings = detunings
        self.threshold = threshold
        
        # Info that needs to be accessible for adiabatic 'substitution'
        self.levels_to_eliminate = {} # ion_indx (int) -> levels_to_eliminate set(int)
        self.new_two_level_systems = {} # lvl pair tuple key> eff rabi, D(alpha) tuple value
        self.old_two_level_system_info = {'lasers': dict(), 'mot_terms': dict()}
        self.transformations = {}


        # Iterate over pairs of rabi frequencies
        ion_indices = sorted(list(set([indx for indx,_,_,_ in rabi_freqs])))
        laser_indices = sorted(list(set([indx for _,indx,_,_ in rabi_freqs])))

        for ion_indx in ion_indices:

            self.new_two_level_systems[ion_indx] = {}

            self.old_two_level_system_info['mot_terms'][ion_indx] = {}

            for laser_indx in laser_indices:                

                # For a given ion and laser index selection, retrieve the compatible i,j indices
                ij_pairs = []
                for ion_indx_inner, laser_indx_inner, i, j in rabi_freqs:
                    if ion_indx == ion_indx_inner and laser_indx == laser_indx_inner:
                        if (i,j) not in ij_pairs:
                            ij_pairs.append((i,j),)

                # Assuming this is only a three level system
                rabi1 = self.rabis[(ion_indx, laser_indx, *ij_pairs[0])]
                rabi2 = self.rabis[(ion_indx, laser_indx, *ij_pairs[1])]

                Delta = self.detunings[(ion_indx, laser_indx, *ij_pairs[1])]

                if not Delta:
                    continue

                rabi_eff = rabi1*rabi2/(2*Delta)

                if rabi_eff/Delta < self.threshold:
                    
                    # Track the level they have in common (to eliminate), as well as the ion it belongs to!
                    common_lvl = list(set(ij_pairs[0]).intersection(set(ij_pairs[1])))[0]
                    if ion_indx in self.levels_to_eliminate:
                        self.levels_to_eliminate[ion_indx].add(common_lvl)
                    else:
                        self.levels_to_eliminate[ion_indx] = {common_lvl}
                    
                    # Eliminating this level will result in an effective new 2-level systems

                    remaining_lvls = set(ij_pairs[0]).symmetric_difference(set(ij_pairs[1]))

                    unique_pairs = list(itertools.combinations(list(remaining_lvls), 2))
                    unique_pairs = [sorted(pair, reverse=True) for pair in unique_pairs]

                    for i,j in unique_pairs:

                        self.transformations[ion_indx] = {(i,j): ij_pairs}

                        if (i,j) in self.new_two_level_systems[ion_indx]:
                            # Only keep the larger of the two effective Rabi frequencies
                            # There are two pairs of possible effective Rabi frequencies because have 2 transitions and 2 lasers

                            if rabi_eff > self.new_two_level_systems[ion_indx][(i,j)]['rabi_eff']:
                                self.new_two_level_systems[ion_indx][(i,j)]['rabi_eff'] = rabi_eff
                                self.new_two_level_systems[ion_indx][(i,j)]['lasers'].append(laser_indx)
                            else:
                                continue
                        else:                   
                            self.new_two_level_systems[ion_indx][(i,j)] = {'rabi_eff': rabi_eff, 'lasers': [laser_indx]}



    def map_OperatorMul(self, model):

        op1 = model.op1
        op2 = model.op2

        if isinstance(op1, OperatorScalarMul) and isinstance(op2, OperatorKron):
            
            phase = op1.coeff.phase

            mot_term = op2

            ic = op1.coeff
            
            i, j = ic.i, ic.j
            ion_indx, laser_indx = ic.ion_indx, ic.laser_indx

            if (ion_indx in self.levels_to_eliminate and (i in self.levels_to_eliminate[ion_indx]
                or j in self.levels_to_eliminate[ion_indx])):

                self.old_two_level_system_info['lasers'][laser_indx] = {'phase': phase}

                mot_terms_n = self.old_two_level_system_info['mot_terms'][ion_indx]

                if laser_indx in mot_terms_n:
                    mot_terms_n[laser_indx][(i,j)] = mot_term
                    
                else:
                    mot_terms_n[laser_indx] = {(i,j): mot_term}

                return Zero()
                
        else:
            return model



class SimplifyZero(RewriteRule):

    def map_OperatorAdd(self, model):

        op1 = model.op1
        op2 = model.op2

        if isinstance(op1, Zero) and not isinstance(op2, Zero):
            return op2

        elif isinstance(op2, Zero) and not isinstance(op1, Zero):
            return op1
        elif isinstance(op1, Zero) and isinstance(op2, Zero):
            
            return Zero()
        else:
            return model



class ReplaceKetBra(RewriteRule):

    def __init__(self, i, j):
        self.i = i
        self.j = j 

    def map_KetBra(self, model):
        return KetBra(i = self.i, j = self.j, dims = model.dims)

def extended_kron(op_list):
    '''
    Given a list of operators, use the OperatorKron operator to
    tensor them all together
    '''

    if len(op_list) == 2:
        return OperatorKron(op1 = op_list[0], op2 = op_list[1])
    
    return extended_kron([op_list[0], extended_kron(op_list[1:])])

from math import cos, sin, sqrt, atan

def simplify_complex_subtraction(wc1, wc2):

    A = wc1.amplitude
    B = wc2.amplitude

    freqA = wc1.frequency
    freqB = wc2.frequency

    if freqA != freqB:
        raise ValueError("Helper function expects both frequencies to be the same")

    phiA = wc1.phase
    phiB = wc2.phase


    re = A*cos(phiA) - B*cos(phiB)
    im = A*sin(phiA) - B*sin(phiB)

    C = sqrt(re**2 + im**2)

    phiC = atan(im/re)

    return WaveCoefficient(amplitude = C,
                           frequency = freqA,
                           phase = phiC,
                            ion_indx = None,
                            laser_indx = None,
                            mode_indx = None,
                            i = None,
                            j = None)

def substitute(chopped_tree, old_info, two_level_info, N, L, ion_N_levels, mode_cutoffs, detunings, transformations):

    print("Transformations:")

    for ion_indx in transformations:
        trans_dict = transformations[ion_indx]
        print(f'{list(trans_dict.values())[0][0]} -> {list(trans_dict.keys())[0]} in ion {ion_indx}.')


    new_tree = chopped_tree

    # First iterate over the different ions
    for ion_indx in two_level_info:

        mot_terms_n = old_info['mot_terms'][ion_indx]



        # For each ion, there may be potentially many new 2-level systems.
        # Construct and add these terms to the chopped tree
        
        J_n = ion_N_levels[ion_indx]# number of levels in current ion

        for i,j in two_level_info[ion_indx]:

            rabi_eff = two_level_info[ion_indx][(i,j)]['rabi_eff']

            
            phi_eff = 0
            for m, laser_indx in enumerate(two_level_info[ion_indx][(i,j)]['lasers']):
                if not m:
                    phi_eff += old_info['lasers'][laser_indx]['phase']
                else:
                    phi_eff -= old_info['lasers'][laser_indx]['phase']

    


            # With i,j, N, L and the ion_indx, I have all necessary info. to construct the internal state term: int_term

            int_coup_int_buffer = [Identity(dims = ion_N_levels[n]) for n in range(N)]
            int_coup_int_buffer[ion_indx] = KetBra(i = i, j = j, dims = J_n)

            int_coup_mot_buffer = [Identity(dims = mode_cutoffs[l]) for l in range(L)]

            new_int_term = extended_kron(int_coup_int_buffer + int_coup_mot_buffer)


            eliminated_two_level_pairs = transformations[ion_indx][(i,j)]
            
            alphas = {}
            alphas_eff = [] # list for when you consider more than 1 mode

            for laser_indx in two_level_info[ion_indx][(i,j)]['lasers']:
                alphas[laser_indx] = extract_alphas(mot_terms_n[laser_indx][eliminated_two_level_pairs[0]])

            for m, laser_indx in enumerate(alphas):
                for a, alpha in enumerate(alphas[laser_indx]):
                    if not m:
                        alphas_eff.append(alpha)
                    else:
                        alphas_eff[a] = simplify_complex_subtraction(alphas_eff[a], alpha)


            mot_coup_int_buffer = [Identity(dims = ion_N_levels[n]) for n in range(N)]

            mot_coup_mot_buffer = [Identity(dims = mode_cutoffs[l]) for l in range(L)]

            for a, alpha_eff in enumerate(alphas_eff):
                mot_coup_mot_buffer[a] = Displacement(alpha = alpha_eff, dims = mode_cutoffs[a])

            new_mot_term = extended_kron(mot_coup_int_buffer + mot_coup_mot_buffer)

            # Need to generalize these hard coded indices; currently assume that levels are in ascending order of energy
            delta_eff = detunings[(ion_indx, 0, 2, 0)] - detunings[(ion_indx, 1, 2, 1)]



            rabi_prefactor = WaveCoefficient(amplitude = rabi_eff/2,
                                             frequency = delta_eff,
                                             phase = phi_eff,
                                             ion_indx = None,
                                             laser_indx = None,
                                             mode_indx = None,
                                             i = None,
                                             j = None)
            
            new_two_level_tree = OperatorMul(op1 = OperatorScalarMul(coeff = rabi_prefactor, op = new_int_term), 
                                        op2 = new_mot_term)
            
    
            new_tree += new_two_level_tree


    return new_tree


'''Pure in the sense that it doesn't keep track of info needed later for substitution'''
class PureElimination(RewriteRule):

    def __init__(self, rabi_freqs, detunings, threshold = 1e-2):
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

            P = rabi**2/Delta**2
            self.transition_probs[tup_key] = P
            if P < self.threshold:
                self.weakly_coupled_transtions.append((ion_indx, i,j),)


        print("WEAKLY COUPLED TRANSITIONS:", self.weakly_coupled_transtions)
        print("TRANSITION PROBS:", self.transition_probs)

    def map_OperatorMul(self, model):

        op1 = model.op1
        op2 = model.op2

        if isinstance(op1, OperatorScalarMul) and isinstance(op2, OperatorKron):
            
            ic = op1.coeff
            
            i, j = ic.i, ic.j
            ion_indx, _ = ic.ion_indx, ic.laser_indx

            if (ion_indx, i, j) in self.weakly_coupled_transtions:
                return Zero()

def simple_adiabatic_elimination(tree, threshold):

    get_freqs = GetRabiFrequenciesDetunings()

    _ = Post(get_freqs)(tree)

    rabis = get_freqs.rabis
    detunings = get_freqs.detunings


    eliminate = PureElimination(rabi_freqs = rabis, detunings = detunings, threshold=threshold)
    eliminator = Post(eliminate)
    chopped_tree = eliminator(tree)

    simplifier = FixedPoint(Post(SimplifyZero()))

    return simplifier(chopped_tree)


def adiabatic_elimination(H, threshold):

    tree = H.tree
    N, L, ion_N_levels, mode_cutoffs = H.N, H.L, H.ion_N_levels, H.mode_cutoffs

    get_freqs = GetRabiFrequenciesDetunings()

    _ = Post(get_freqs)(tree)

    rabis = get_freqs.rabis
    detunings = get_freqs.detunings


    eliminate = Elimination(rabi_freqs = rabis, detunings = detunings, threshold=threshold)
    eliminator = Post(eliminate)
    chopped_tree = eliminator(tree)

    final_tree = substitute(chopped_tree=chopped_tree, old_info = eliminate.old_two_level_system_info, 
                            two_level_info=eliminate.new_two_level_systems, N = N, L = L, ion_N_levels = ion_N_levels, 
                            mode_cutoffs = mode_cutoffs, detunings = detunings, transformations = eliminate.transformations)

        
    simplifier = FixedPoint(Post(SimplifyZero()))

    final_tree = simplifier(final_tree)

    return final_tree



    