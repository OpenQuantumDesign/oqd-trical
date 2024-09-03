import numpy as np
from math import acos, sin, sqrt, cos
from sympy.physics.wigner import wigner_3j, wigner_6j
import trical.misc.constants as cst
from trical.light_matter.utilities import zeeman_energy_shift

class Chamber():
    '''Class summarizing experimental description
    
    Args:
        chain (LinearChain): LinearChain object housing ions and their modes
        B (float): magnetic field magnitude
        Bhat (iterable): magnetic field direction
        lasers (list[Laser]): list of Laser objects
    '''
    def __init__(self, chain, B, Bhat, lasers):

        self.chain = chain
        self.B = B
        self.Bhat = Bhat
        self.lasers = lasers

        # Correct energy level splittings accounting for Zeeman effect

        for ion in self.chain.ions:
            for alias in ion.levels:
                level = ion.levels[alias]
                level.energy += zeeman_energy_shift(level, self.B)/(cst.hbar*2*np.pi)

    def set_laser_wavelength_from_transition(self, laser_index, transition):
        '''Method for setting laser wavelength to the transition wavelength of a transition        
        
        Note: 
            .wavelength attribute of laser with specified laser index mutated; set to wvl

        Args:
            laser_index (int): 0-indexed integer pointing to the laser user would like set wavelength for.
                             Order based on chamber instantiation (index of laser in self.lasers)
            transition (Transition): transition object you'd like to set the laser resonant on 
        Returns:
            wvl (float): wavelength which the laser has been set to
        '''

        lvl1, lvl2 = transition.level1, transition.level2

        f1, f2 = lvl1.energy, lvl2.energy

        wvl = cst.c/abs(f2-f1)
        self.lasers[laser_index].wavelength = wvl
        return wvl

    def set_laser_intensity_from_pi_time(self, laser_index, pi_time, transition):
        '''Method for setting laser intensity such that resonant pi time is the one specified
        
        Note: 
            .I and .I_0 atttributes of laser with specified laser index mutated to resulting intensity
        
        Args:
            laser_index (int): 0-indexed integer pointing to the laser user would like set wavelength for.
                             Order based on chamber instantiation.
            pi_time (float): pi time when (this single) laser resonant on transition
            transition (Transition): transition object 
        
        Returns:
            intensity_fn (function): intensity (constant) function of time

        '''

        laser = self.lasers[laser_index]

        matrix_elem = self.compute_matrix_element(laser, transition)

        I =  cst.eps_0 * cst.c/2 * ((cst.hbar*np.pi)/(pi_time * matrix_elem * cst.e))**2
        
        def intensity_fn(t):
            return I
        laser.I = intensity_fn
        laser.I_0 = I
        return intensity_fn

    def set_laser_intensity_from_rabi_frequency(self, laser_index, rabi_frequency, transition):
        '''Method for setting laser intensity such that resonant rabi frequency is the one specified
        
        Note: 
            .I and .I_0 atttributes of laser with specified laser index mutated to resulting intensity
        
        Args:
            laser_index (int): 0-indexed integer pointing to the laser user would like set wavelength for.
                             Order based on chamber instantiation.
            rabi_frequency (float): rabi frequency when (this single) laser resonant on transition
            transition (Transition): transition object 
        
        Returns:
            intensity_fn (function): intensity (constant) function of time

        '''
        
        pi_time = np.pi/(2*np.pi*rabi_frequency)

        # Remember set_laser_intensity_from_pi_time mutates the laser's I, I_0 in place
        intensity_fn = self.set_laser_intensity_from_pi_time(laser_index, pi_time, transition)

        return intensity_fn
    
    def rabi_frequency_from_intensity(self, laser_index, intensity, transition):
        ''' Method computing a transition's resonant rabi frequency addressed by a laser and its intensity

        Args:
            laser_index (int): 0-indexed integer pointing to the laser user would like set wavelength for.
                             Order based on chamber instantiation.
            intensity (float): laser intensity in W/m^2
            transition (Transition): transition object
        Returns:
            rabi_frequency (float)
        '''

        laser = self.lasers[laser_index]

        matrix_elem = self.compute_matrix_element(laser, transition)
        E = sqrt(2*intensity/(cst.eps_0*cst.c)) # Formula for electric field magnitude given intensity

        return matrix_elem * E * cst.e / cst.hbar # Definition of Rabi frequency
    
    def compute_matrix_element(self, laser, transition):
        '''Method that compute dipole and quadrupole matrix elements 

        Args:
            laser (Laser): laser object for accessing polarization and wavevector information
            transition (Transition): transition object for accessing quantum number of levels
        
        Returns:
            (float): Multipole matrix elements for E1, E2 transitions
        '''
        
        lvl1, lvl2 = transition.level1, transition.level2

        J1, J2 = lvl1.spin_orbital, lvl2.spin_orbital
        F1, F2 = lvl1.spin_orbital_nuclear, lvl2.spin_orbital_nuclear
        M1, M2, = lvl1.spin_orbital_nuclear_magnetization, lvl2.spin_orbital_nuclear_magnetization

        q = M2 - M1

        eps_hat = np.array(laser.eps_hat).T # make it a column vector
        k_hat = laser.k_hat

        # If the B field is not aligned with the axial direction, perform a rotation R on the laser
        # wavevector and polarization. 

        if not np.array_equal(self.Bhat, np.array([0,0,1])):
            # rotate these units vectors

            a  = np.cross(self.Bhat, np.array([0,0,1]))
            a /= np.linalg.norm(a)
            theta = acos(self.Bhat[2])
            amatrix = np.array([
                [0, -a[2], a[1]],
                [a[2], 0, -a[0]],
                [-a[1], a[0], 0]
            ])
            R = np.identity(3) + sin(theta) * amatrix + 2*sin(theta/2)**2 * np.matmul(amatrix, amatrix)

            eps_hat = np.matmul(R, eps_hat)
            k_hat = np.matmul(R, k_hat)



        E1, E2 = lvl1.energy, lvl2.energy

        # Just by convention; these orderings are set upon instantiation of an Ion object
        if E1 > E2:
            raise ValueError("Expected E2 > E1")

        # If this happens there's probably an error with the ion species card
        if lvl1.nuclear != lvl2.nuclear:
            raise ValueError("Different nuclear spins between two levels in transition:", transition)
        I = lvl1.nuclear
        
        A = transition.einsteinA
        omega_0 = 2*np.pi* (E2 - E1)

        if transition.multipole == "E1":

            units_term = sqrt((3*np.pi*cst.eps_0*cst.hbar*cst.c**3)/omega_0 * A) / (omega_0*cst.e) #<- anomalous constants I needed to add... hmm
            hyperfine_term = sqrt((2*F2+1)*(2*F1+1)) * wigner_6j(J1, J2, 1, F2, F1, I)

            # q -> polarization
            polarization_map = {-1: 1/sqrt(2) * np.array([1, 1j, 0]), 
                                0: np.array([0,0,1]),
                                1: 1/sqrt(2) * np.array([1, -1j, 0])}


            geometry_term = sqrt(2*J2+1) * polarization_map[q].dot(eps_hat) * wigner_3j(F2, 1, F1, M2, -q, -M1)
            
            return abs(units_term) * abs(geometry_term) * abs(hyperfine_term)
    
        elif transition.multipole == "E2":

            units_term = sqrt((15*np.pi*cst.eps_0*cst.hbar*cst.c**3)/omega_0 * A) / (omega_0*cst.e) #<- anomalous constants I needed to add... hmm
            hyperfine_term = sqrt((2*F2+1)*(2*F1+1)) * wigner_6j(J1, J2, 2, F2, F1, I)

            # q -> polarization
            polarization_map = {-2: 1/sqrt(6) * np.array([[1, 1j, 0],
                                                            [1j, -1, 0],
                                                            [0, 0, 0]]),
                                -1: 1/sqrt(6) * np.array([[0, 0, 1],
                                                            [0, 0, 1j],
                                                            [1, 1j, 0]]),
                                0: 1/3 * np.array([[-1, 0, 0],
                                                    [0, -1, 0],
                                                    [0, 0, 2]]),
                                1: 1/sqrt(6) * np.array([[0, 0, -1],
                                                            [0, 0, 1j],
                                                            [-1, 1j, 0]]),
                                2: 1/sqrt(6) * np.array([[1, -1j, 0],
                                                            [-1j, -1, 0],
                                                            [0, 0, 0]])
                                }

            geometry_term = sqrt(2*J2+1) * k_hat.dot(np.matmul(polarization_map[q], eps_hat)) * wigner_3j(F2, 2, F1, M2, -q, -M1)

            return abs(units_term) * abs(geometry_term) * abs(hyperfine_term)   
            
        else:
            raise ValueError("Currently only support dipole and quadrupole allowed transitions")
