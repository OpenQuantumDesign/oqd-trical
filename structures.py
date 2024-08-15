from qutip import basis, Qobj, tensor, qeye, destroy
import numpy as np
import trical

from scipy.special import genlaguerre as L
from math import factorial, sqrt, exp
from numbers import Number


class VibrationalMode():
    def __init__(self, eigenfreq, eigenvect, axis, N):
        
        self.eigenfreq = eigenfreq
        self.eigenvect = eigenvect
        
        self.axis = axis 
        self.N = N       

        
    def setstate(self, n):
        if n >= self.N:
            raise ValueError("Outside of Hilbert space")
        return basis(self.N, n)
    
    def groundstate(self):
        return basis(self.N, 0)
    
    def modecutoff(self, val):
        self.N = val + 1


    """
    Computes matrix elements of displacement operator in the number basis
    """

    def D_mn(self, m, n, alpha):

        if m >= n:
            out = sqrt(factorial(n)/factorial(m)) * alpha ** (m-n) * exp(-1/2 * abs(alpha)**2) * \
            L(n = n, alpha = m-n)(abs(alpha)**2)
        else:
            out = sqrt(factorial(m)/factorial(n)) * (-alpha.conjugate()) ** (n-m) * exp(-1/2 * abs(alpha)**2) * \
            L(n = m, alpha = n-m)(abs(alpha)**2)
  
        return out

    def displace(self, alpha, ld_order, rwa_cutoff, nu, Delta):
        
        op = [[0 for _ in range(self.N)] for _ in range(self.N)]
        
        for m in range(0, self.N):
            for n in range(0, self.N):

                if abs(m-n) > ld_order:
                    op[m][n] = 0
                elif rwa_cutoff != 'inf':
                    if abs((m-n)*nu - Delta) > rwa_cutoff:
                        op[m][n] = 0
                    else:
                        op[m][n] = self.D_mn(m, n, alpha)
                else:
                    op[m][n] = self.D_mn(m, n, alpha)

        out = Qobj(op)

        return out

class Laser():
    def __init__(self, wavelength = None, k_hat = None, I = None, eps_hat = [0,0,1], phi = 0):
        self.wavelength = wavelength
        self.phi = phi

        if isinstance(I, Number):
            # set as constant function
            def intensity_fn(t):
                return I
            self.I = intensity_fn
        else:
            self.I = I
        
        
        self.eps_hat = eps_hat
        self.k_hat = k_hat
        self.detuning = 0

    def detune(self, Delta):
        self.detuning = Delta


'''
TODO: CURRENTLY ASSUMES IONS OF THE SAME SPECIES
'''
class Chain():
    def __init__(self, ions, trap_freqs, selected_modes):
        
        self.ions = ions
        self.trap_freqs = trap_freqs # w_x, w_y, w_z
        self.selected_modes = selected_modes
        
        N = len(ions)
        mass = ions[0].mass

        omega_x = 2 * np.pi * trap_freqs[0]
        omega_y = 2 * np.pi * trap_freqs[1]
        omega_z = 2 * np.pi * trap_freqs[2]
        
        alpha = np.zeros((3, 3, 3))
        alpha[2, 0, 0] = mass * (omega_x) ** 2 / 2
        alpha[0, 2, 0] = mass * (omega_y) ** 2 / 2
        alpha[0, 0, 2] = mass * (omega_z) ** 2 / 2

        pp = trical.classes.PolynomialPotential(alpha, N=N) # polynomial potential
        ti = trical.classes.TrappedIons(N, pp, m = mass)
        ti.principle_axis()
        
        eigenfreqs = ti.w_pa / (2 * np.pi) # make frequencies available to users linear
        eigenvects = ti.b_pa

        self.modes = []

        for l in range(len(eigenfreqs)):
            if 0 <= l <= N-1:
                axis = np.array([1,0,0])
            elif N <= l <= 2*N-1:
                axis = np.array([0,1,0])
            elif 2*N <= l <= 3*N-1:
                axis = np.array([0,0,1])
            else:
                raise ValueError("Freq direction sorting went wrong :(")

            self.modes.append(VibrationalMode(eigenfreqs[l], eigenvects[l], axis, 10))

        self.eqm_pts = ti.equilibrium_position()

    def ion_projector(self, ion_numbers, names):
        """
        Full Hilbert space projector onto internal state "name" of ion "ion_number"

        Parameters:
        - ion_number (int) or ion_numbers (list(int)): list or single ion identifier
        - name (str) or names (list(str)): list or single label/alias for a ion's internal state
        -- 
        """

        mot_buffer = [qeye(mode.N) for mode in self.selected_modes]
        

        if type(names) == str and type(ion_numbers) == int:
             # Only one name was provided; single-ion case
            ion_buffer = [qeye(ion.N_levels) for ion in self.ions]
            
            ket = self.ions[ion_numbers-1].state[names]
            ion_buffer[ion_numbers-1]  = ket * ket.dag()# 0-indexed, place projector
            name = names
        if type(names) == list:
            # Multiple names were provided; multi-ion case
            ket = tensor(*[self.ions[ion_numbers[j]-1].state[names[j]] for j in range(len(names))])
            ion_buffer = []
            for j in range(len(self.ions)):
               
                # If ion isn't being projected onto, place identity
                if j+1 not in ion_numbers:
                    ion_buffer.append(qeye(self.ions[j].N_levels))
                
                # If ion is the *first* (of multiple) being projected onto, insert the projector
                # Otherwise, continue 
                elif j+1 == ion_numbers[0]:
                    ion_buffer.append(ket * ket.dag())
            name = ''.join(names)

        return QuantumOperator(qobj=tensor(*ion_buffer, *mot_buffer), name = name)
    
    def number_operator(self, mode_indx, name):

        ion_buffer = [qeye(ion.N_levels) for ion in self.ions]
        mot_buffer = [qeye(mode.N) for mode in self.selected_modes]

        dims = self.selected_modes[mode_indx].N
        mot_buffer[mode_indx] = destroy(N = dims).dag() * destroy(N = dims)


        return QuantumOperator(qobj = tensor(*ion_buffer, *mot_buffer), name = name)


class QuantumOperator():
    def __init__(self, qobj, name):
        self.qobj = qobj
        self.name = name




