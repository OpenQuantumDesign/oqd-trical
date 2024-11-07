import numpy as np
from math import sqrt, factorial
from cmath import exp
from scipy.special import genlaguerre as L
from typing_extensions import Annotated
from qutip import Qobj, QobjEvo
from pydantic import AfterValidator, NonNegativeFloat
from ..misc import constants as cst

# OPERATOR UTILITIES

def displace(ld_order, alpha, rwa_cutoff, Delta, nu, dims):
    """Constructs time-dependent, Qutip-compatible displacement operator

    Args:
        ld_order (int): Lamb-Dicke approximation order
        alpha (time-dependent function): coherent state parameter of mode
        rwa_cutoff (float): user-specified cutoff for oscillating terms
        Delta (float): detuning in scaled (by timescale) angular frequency
        nu (float): mode eigenfrequency in scaled (by timescale) angular frequency
        dims (int): phonon cutoff for mode

    Returns:
        (QobjEvo): Time-dependent displacement operator that's dims x dims in shape
    """

    def time_dep_fn(t, args={}):

        evaled_alpha = alpha(t=t, args=args)

        # Construct displacement matrix
        op = [[0 for _ in range(dims)] for _ in range(dims)]

        for m in range(0, dims):
            for n in range(0, dims):

                if abs(m - n) > ld_order:
                    op[m][n] = 0
                elif rwa_cutoff != "inf":
                    # For explanation on RWA condition, check Documentation Supplement, RWA section
                    if abs((m - n) * nu - Delta) > rwa_cutoff:
                        op[m][n] = 0
                    else:
                        op[m][n] = D_mn(m, n, evaled_alpha)
                else:
                    op[m][n] = D_mn(m, n, evaled_alpha)
        return Qobj(op)
    return QobjEvo(time_dep_fn)

# def detune(self, Delta):
#         """Set the laser's detuning.

#         Args:
#             Delta (float): The detuning value to set for the laser.
#         """
#         self.detuning = Delta

# MOVE:
def lambdicke(mode, ion, laser):
    """Helper function for computing the Lamb-Dicke parameter $\\eta$

    Args:
        mode (Mode): Mode object; used to access the mode's eigenfrequency and axis
        ion (Ion): Ion object; used to access the ion's mass
        laser (Laser): Laser object; used to access the laser's wavevector info

    Returns:
        (float): lamb-dicke parameter
    """

    x0 = sqrt(cst.hbar / (2 * ion.mass * 2 * np.pi * mode.eigenfreq))
    k = 2 * np.pi / laser.wavelength
    return x0 * k * laser.k_hat.dot(mode.axis)

#     return Results(qutip_results=out, ops=expt_ops, times=times, timescale=timescale)

def D_mn(m, n, alpha):
    """Computes matrix elements of displacement operator in the number basis

    Args:
        m (int): matrix element row/bra index
        n (int): matrix element column/ket index
        alpha (complex float): alpha evaluated at some time t such that it is not longer a function of t,
                             but rather a complex float

    Returns:
        (float): displacement operator matrix element D_mn
    """

    if m >= n:
        out = (
            sqrt(factorial(n) / factorial(m))
            * alpha ** (m - n)
            * exp(-1 / 2 * abs(alpha) ** 2)
            * L(n=n, alpha=m - n)(abs(alpha) ** 2)
        )
    else:
        out = (
            sqrt(factorial(m) / factorial(n))
            * (-alpha.conjugate()) ** (n - m)
            * exp(-1 / 2 * abs(alpha) ** 2)
            * L(n=m, alpha=n - m)(abs(alpha) ** 2)
        )
    return out

# Args are t and args = {} because this will be cast the QobjEvo

def to_dm(ensemble):
    """Constructs a density matrix from an ensemble of states

    Args:
        ensemble (List[Tuple[Qobj, float]] OR Qobj): takes in a list of ket (Qobj), probability tuple pairs
        OR a single Qobj, with probability assume to be 1
    Returns:
        (Qobj): density matrix
    """
    if type(ensemble) == Qobj:
        ket = ensemble
        return ket * ket.dag()
    out = 0

    for ket, prob in ensemble:
        out += prob * ket * ket.dag() 
    return out

def is_halfint(v: float) -> float:
    """
    Checks if a value is a half-integer.

    Args:
        v (float): The value to check.

    Returns:
        float: The input value if it is a half-integer.

    Raises:
        ValueError: If the value is not a half-integer.
    """
    if not (v * 2).is_integer():
        raise ValueError(f"Value {v} is not a half-integer.")
    return v

AngularMomentumNumber = Annotated[float, AfterValidator(is_halfint)]
NonNegativeAngularMomentumNumber = Annotated[
    NonNegativeFloat, AfterValidator(is_halfint)
]

# def compute_zeeman_levels(lvl, B):
#     '''
#     Parameters:
#     - lvl (Level): Electronic level in the absence of a magnetic fieldreturn the Zeeman-split levels
#     - B (float): magnitude of magnetic field in the chamber

#     Returns:
#     - zeeman_lvls (list[Levels]): Zeeman level list ordered by increasing magnetization number
#     '''

#     freq = lvl.energy # actually contains linear frequency
#     F = lvl.spin_orbital_nuclear
#     J = lvl.spin_orbital
#     L = lvl.orbital

#     zeeman_lvls = []

#     for m_F in np.arange(-F, F + 1, 1):

#         freq_prime = freq + zeeman_energy_shift(L, J, m_F, B)/cst.h # perturbed frequency

#         zeeman_lvls.append(
#             Level(
#                 principle = lvl.prinicpal,
#                 spin = lvl.spin,
#                 orbital = L,
#                 nuclear = lvl.nuclear,
#                 spin_orbital = J,
#                 spin_orbital_nuclear = F,
#                 spin_orbital_nuclear_magnetization = m_F,
#                 energy = freq_prime)
#                 )
#     return zeeman_lvls