"""SUMMARY
"""
import numpy as np


class SpinLattice(object):

    """SUMMARY
    
    Attributes:
        J (TYPE): DESCRIPTION
    """
    
    def __init__(self, J):
        """SUMMARY
        
        Args:
            J (TYPE): DESCRIPTION
        """
        super(SpinLattice, self).__init__()

        self.J = J
        pass

    pass


class SimulatedSpinLattice(SpinLattice):

    """SUMMARY
    
    Attributes:
        ic (TYPE): DESCRIPTION
        mu (TYPE): DESCRIPTION
        Omega (TYPE): DESCRIPTION
    """
    
    def __init__(self, ic, mu, Omega, **kwargs):
        """SUMMARY
        
        Args:
            ic (TYPE): DESCRIPTION
            mu (TYPE): DESCRIPTION
            Omega (TYPE): DESCRIPTION
            **kwargs: DESCRIPTION
        """
        self.ic = ic
        self.mu = mu
        self.Omega = Omega

        super(SimulatedSpinLattice, self).__init__()
        pass

    pass
