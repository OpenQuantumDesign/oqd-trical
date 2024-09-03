from trical.light_matter.utilities import lambdicke
from qutip import basis, qeye, tensor, QobjEvo, destroy
from cmath import exp
import numpy as np
import trical.misc.constants as cst

'''
NOW DEPRECATED; was the file used for constructing Hamiltonian QobjEvo (not tree first)
'''

def construct_H(chamber, timescale, ld_order = 1, rwa_cutoff = 'inf', boost_freq = None, sym_rep = False):
    
    """
    Constructs system Hamiltonian in the interaction picture. For more information about
    how this Hamiltonian was derived please visit this Overleaf document:
    
    https://www.overleaf.com/read/dptrvjrtbjbf#4b0ace


    Parameters:
    - chamber: chamber/system structure 
    - timescale (float): float representing time unit; e.g. 10e-6 for micro-seconds
    - ld_order (int): Lamb-Dicke approximation cutoff when constructing displacement operator
    - rwa_cutoff ('inf' or float): omit all terms in H matrix that oscillate faster than linear frequency 'rwa_cutoff'.
                                   checked for in construction of displacement operator
    - sym_rep (bool): return 
    """
    
    chain = chamber.chain
    lasers = chamber.lasers

    ions = chamber.chain.ions

    if rwa_cutoff != 'inf':
        rwa_cutoff *= 2*np.pi*timescale
    
    def time_dep_H(t, args):

        H, H_sym = 0, ''


        for m, laser in enumerate(lasers):

            for n, ion in enumerate(ions):

                J_n = ion.N_levels

                for transition in ion.transitions:
                    
                    lbl1, lbl2, = transition
                    ket1, ket2 = ion.state[lbl1], ion.state[lbl2]
                    bra1, bra2 = ket1.dag(), ket2.dag()

                    lvl1, lvl2  = ion.transitions[transition].level1, ion.transitions[transition].level2


                    I = laser.I(t) # intensity generally some function of time to be evaluated at each time step
                    
                    Omega_nm =  chamber.rabi_frequency_from_intensity(laser_index= m, intensity = I, 
                                                                      transition = ion.transitions[transition]) * timescale
                            
                    
                    # Construct first term in Hamiltonian

                    # Already accounted for Zeeman shift when constructing chamber
                    
                    f = abs(lvl2.energy - lvl1.energy)

                    omega_0 = 2*np.pi*f * timescale # transition frequency

                    laser_freq = 2*np.pi* (cst.c / laser.wavelength + laser.detuning)*timescale
                    
                    Delta_nmkj = laser_freq - omega_0 # no need to multiply by timescale since laser_freq and omega_0 already scaled!
                    
                    # Atomic transition part:
                    # Aren't I assuming here that all ions have the same number of levels?
                    int_buffer = [qeye(J_n) for _ in range(len(ions))]

                    sigma_p = ket2*bra1

                    int_buffer[n] = sigma_p

                    # Full Hilbert space raising operator 
                    sigma = tensor(*int_buffer, *[qeye(mode.N) for mode in chain.selected_modes])

                    int_term = sigma * exp(-1j*(Delta_nmkj*t - laser.phi))

                    #Delta_nmkj_sym = f"\Delta_{{{f'{n+1},{m+1},{lbl1},{lbl2}'}}}"
                    
                    
                    Delta_nmkj_sym = f"\left(\omega_{m+1} - " + f"\omega^{{{f'{lbl1},{lbl2}'}}}_{{{f'0,{n+1}'}}}" + r'\right)'
                    exp_arg_sym = f"-i\left[{{{Delta_nmkj_sym} t - \phi_{m+1}}}" + r'\right]'
                    int_term_sym = rf'|{lbl2}\rangle \langle {lbl1}|' + f"e^{{{f'{exp_arg_sym}'}}}"
                    
                    #f"\exp(-i\left[{{{Delta_nmkj_sym} t - \phi_{m+1}}})" + r'\right]]'

                    # Motional mode part:

                    mot_term = 1
                    mot_term_sym = ''
                    for l, mode in enumerate(chain.selected_modes):
                        # Define annihilation and creation operators based on phonon cutoff and number of modes and ions
                        nu = 2*np.pi*mode.eigenfreq * timescale # mode eigenfrequency
                        
                        eta_nml = lambdicke(mode, ion, laser)

                        alpha = 1j * eta_nml * exp(1j*nu*t) # coherent state parameter

                        # Define displacement wrt whole Hilbert space
                        mot_buffer = [qeye(mode.N) for mode in chain.selected_modes]
                        mot_buffer[l] = mode.displace(alpha, ld_order, rwa_cutoff, nu, Delta_nmkj)
                        
                        displace = tensor(*[qeye(J_n) for _ in range(len(ions))], *mot_buffer)
                        mot_term *= displace
                        mot_term_sym += rf'D(\alpha_{l+1})'
                    

                    inner_ion_H = Omega_nm/2 * int_term * mot_term
                    inner_ion_H_sym = r'\frac{\hbar}{2}' + f"\Omega_{{{f'{lbl1},{lbl2}'}}}" + int_term_sym + mot_term_sym
                    #inner_ion_H_sym = rf"\Omega_{{lbl1}, {lbl2}}/2" + int_term_sym + mot_term_sym
                    #inner_ion_H_sym = r'\frac{\Omega_{{{lbl1}}}}{2}' +  int_term_sym + mot_term_sym
                                               
                    # Now allow for boosting in a frame rotating at linear frequency 'boost_freq'
                    if boost_freq:

                        # print(2*np.pi*boost_freq*timescale, Delta_nmkj)
                        # Pauli Z buffer
                        pz_int_buffer = [qeye(J_n) for _ in range(len(ions))]
                        pz_int_buffer[n] = ket2*bra2 - ket1*bra1

                        pauli_z = tensor(*pz_int_buffer, *[qeye(mode.N) for mode in chain.selected_modes])
                        H_boost = 2*np.pi*boost_freq * timescale /2 * pauli_z
                        U = (-1j * t * H_boost).expm()
                        # Append tranformed Hamiltonian terms
                        #print(t)
                        #print(U.dag()*(inner_ion_H + inner_ion_H.dag())*U - (inner_ion_H + inner_ion_H.dag()))
                        H += U.dag()*(inner_ion_H + inner_ion_H.dag())*U
                        H_sym += inner_ion_H_sym + '+'
                    else:
                        H_sym += inner_ion_H_sym
                        H += inner_ion_H + inner_ion_H.dag()
        if sym_rep:
                if boost_freq:
                    print(r'$H = U^{\dag}\biggl\{' + H_sym + r'h.c.\biggr\}U$')
                    print("\nWhere $U = e^{-iH_0 t /\hbar}," + r"H_0 = \frac{\hbar\Delta}{2}\sigma_z$")
                else:   
                    print('$H = ' + H_sym + 'h.c.$')
        return H

    return QobjEvo(time_dep_H) 