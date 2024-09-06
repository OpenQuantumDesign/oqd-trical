from math import sqrt, factorial
from cmath import exp

import numpy as np

from scipy.special import genlaguerre as L

from qutip import mesolve, Qobj, QobjEvo

########################################################################################

from ..misc import constants as cst
from .interface.Ion import Level
from ..backend.qutip.Results import Results

########################################################################################


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


def time_evolve(H, psi_0, times, expt_ops=[], progress_bar=False):
    """Basically a wrapper around QuTiP's master equation solver 'mesolve'

    Args:
        H (Hamiltonian): Hamiltonian object
        psi_0 (Qobj): initial state at as QuTiP compatible object
        times (List[float]): times at which to evaluate the expectation values of operators in expt_ops
        expt_ops (List(QuantumOperators)): list of QuantumOperators as defined in structures.py
        progress_bar (Bool): show progress bar while integrating?

    Returns:
        (Results): Results object as defined in Results.py
    """

    timescale = H.args["timescale"]

    out = mesolve(
        H.convert_to_qutip(),
        psi_0,
        times,
        e_ops=[op.qobj for op in expt_ops],
        args={},
        options={"progress_bar": progress_bar, "store_final_state": True},
    )

    return Results(qutip_results=out, ops=expt_ops, times=times, timescale=timescale)


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

        out += prob * ket * ket.dag()  # weighted sum of projectors by probability

    return out


def zeeman_energy_shift(level, B):
    """Computes the Zeeman energy shift on a level

    Args:
        level (Level): Level object used for accessing quantum numbers
        B (float): magnitude of B-field in chamber

    Returns:
        (float): Zeeman shift in energy of level in B
    """
    L = level.orbital
    J = level.spin_orbital
    F = level.spin_orbital_nuclear
    m_F = level.spin_orbital_nuclear_magnetization

    I = abs(F - J)
    S = abs(J - L)

    # Bohr magneton
    mu_B = 9.27400994e-24

    g_j = 3 / 2 + (S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))

    g_f = (F * (F + 1) + J * (J + 1) - I * (I + 1)) / (2 * F * (F + 1)) * g_j

    return g_f * mu_B * B * m_F


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

        # Convert to static QuTiP-compatible object
        return Qobj(op)

    # Convert to time-dependent QuTiP-compatible object
    return QobjEvo(time_dep_fn)
