"""
Defines the SpinLattice class and its subclasses representing a spin lattice system or a
simulation of a spin lattice system
"""
import numpy as np


class SpinLattice(object):

    """
    Object representing a spin lattice system
    
    Attributes:
        J (2-D array of float): Interaction graph of the system
    """
    
    def __init__(self, J):
        """
        Initialization function for a SpinLattice object
        
        Args:
            J (2-D array of float): Interaction graph of the system
        """
        super(SpinLattice, self).__init__()

        self.J = J
        pass

    pass


class SimulatedSpinLattice(SpinLattice):

    """
    Object representing a spin lattice system simulated by a trapped ion system
    
    Attributes:
        ti (TrappedIons): A trapped ion system
        mu (float or 1-D array of float): Raman beatnote detunings
        Omega (1-D or 2-D array of float): Rabi frequencies
    """
    
    def __init__(self, ti, mu, Omega, **kwargs):
        """
        Initialization function for a SimulatedSpinLattice object
        
        Args:
            ti (TrappedIons): A trapped ion system
            mu (float or 1-D array of float): Raman beatnote detunings
            Omega (1-D or 2-D array of float): Rabi frequencies
        Kwargs:
        """
        self.ti = ti
        self.mu = mu
        self.Omega = Omega

        super(SimulatedSpinLattice, self).__init__()
        pass

    pass
