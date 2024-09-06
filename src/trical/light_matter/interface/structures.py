from numbers import Number

from math import factorial, sqrt, exp

import numpy as np

from scipy.special import genlaguerre as L

from qutip import basis, Qobj, tensor, qeye, destroy

########################################################################################

from ...classes import PolynomialPotential, TrappedIons

########################################################################################


class VibrationalMode:
    """Class representing vibrational mode of an ion chain

    Args:
        eigenfreq (float): eigenfrequency in Hz of the mode
        eigenvect (iterable): eigenvector of mode
        axis (iterable): mode axis of oscillation
        N (int): max phonon cutoff
    """

    def __init__(self, eigenfreq, eigenvect, axis, N):

        self.eigenfreq = eigenfreq
        self.eigenvect = eigenvect

        self.axis = axis
        self.N = N

    def setstate(self, n):
        """Convenience method for accessing mode's nth excited state

        Args:
            n (int): zero-indexed phonon number

        Returns:
            (Qobj): Mode's excited state |n >

        Raises:
            ValueError: if n exceeeds the phonon cutoff

        """
        if n >= self.N:
            raise ValueError("Outside of Hilbert space")
        return basis(self.N, n)

    def groundstate(self):
        """Convenience method for accessing mode ground state

        Returns:
            (Qobj): Mode's groundstate
        """
        return basis(self.N, 0)

    def modecutoff(self, val):
        """Method for setting the mode cutoff

        Note:
            self.N is mutated in place

        Args:
            val (int): new max phonon cutoff
        """

        self.N = val + 1


class Laser:
    """Class for representing a laser

    Args:
        wavelength (float): laser's wavelength
        k_hat (iterable): laser's normalized wave vector
        I (Number or callable): laser's intensity in (W/m^2), either as a constant or function
        eps_hat (iterable): laser's normalized polarization vector
        phi (float): laser's phase

    Attributes:
        detuning (float): how much to detune laser's frequency by in Hz
    """

    def __init__(self, wavelength=None, k_hat=None, I=None, eps_hat=[0, 0, 1], phi=0):
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
        """Method for setting laser's detuning

        Note:
            mutates Laser.detuning in place

        Args:
            Delta (float): detuning in Hz
        """
        self.detuning = Delta


"""
TODO: CURRENTLY ASSUMES IONS OF THE SAME SPECIES
"""


class Chain:
    """Class for representing ion chain

    Args:
        ions (list): list of Ion objects
        trap_freqs (list): list of trap COM frequencies in Hz of the form [omega_x, omega_y, omega_z]
        selected_modes (list): list of Mode objects

    Attributes:
        modes (list): list of Mode objects, now with access to their axis of oscillation
        eqm_pts (list): list of equilibirum positions of the ions

    """

    def __init__(self, ions, trap_freqs, selected_modes):

        self.ions = ions
        self.trap_freqs = trap_freqs  # w_x, w_y, w_z
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

        pp = PolynomialPotential(alpha, N=N)  # polynomial potential
        ti = TrappedIons(N, pp, m=mass)
        ti.principle_axis()

        eigenfreqs = ti.w_pa / (2 * np.pi)  # make frequencies available to users linear
        eigenvects = ti.b_pa

        self.modes = []

        for l in range(len(eigenfreqs)):
            if 0 <= l <= N - 1:
                axis = np.array([1, 0, 0])
            elif N <= l <= 2 * N - 1:
                axis = np.array([0, 1, 0])
            elif 2 * N <= l <= 3 * N - 1:
                axis = np.array([0, 0, 1])
            else:
                raise ValueError("Freq direction sorting went wrong :(")

            self.modes.append(VibrationalMode(eigenfreqs[l], eigenvects[l], axis, 10))

        self.eqm_pts = ti.equilibrium_position()

    def ion_projector(self, ion_numbers, names):
        """Full Hilbert space projector onto internal state "name" of ion "ion_number"

        Args:
            ion_numbers (int or list(int)): list or single ion identifier
            names (str or list(str)): list or single label/alias for a ion's internal state

        Returns:
            (QuantumOperator): QuantumOperator object as defined in structures.py
        """

        mot_buffer = [qeye(mode.N) for mode in self.selected_modes]

        if type(names) == str and type(ion_numbers) == int:
            # Only one name was provided; single-ion case
            ion_buffer = [qeye(ion.N_levels) for ion in self.ions]

            ket = self.ions[ion_numbers - 1].state[names]
            ion_buffer[ion_numbers - 1] = ket * ket.dag()  # 0-indexed, place projector
            name = names
        if type(names) == list:
            # Multiple names were provided; multi-ion case
            ket = tensor(
                *[
                    self.ions[ion_numbers[j] - 1].state[names[j]]
                    for j in range(len(names))
                ]
            )
            ion_buffer = []
            for j in range(len(self.ions)):

                # If ion isn't being projected onto, place identity
                if j + 1 not in ion_numbers:
                    ion_buffer.append(qeye(self.ions[j].N_levels))

                # If ion is the *first* (of multiple) being projected onto, insert the projector
                # Otherwise, continue
                elif j + 1 == ion_numbers[0]:
                    ion_buffer.append(ket * ket.dag())
            name = "".join(names)

        return QuantumOperator(qobj=tensor(*ion_buffer, *mot_buffer), name=name)

    def number_operator(self, mode_indx, name):
        """Full Hilbert space number_operator for Mode object with index mode_indx

        Args:
            mode_indx (int): zero-indexed identifier for mode determined at instantiation
            name (str): alias/name for the number operator

        Returns:
            (QuantumOperator): QuantumOperator object
        """
        ion_buffer = [qeye(ion.N_levels) for ion in self.ions]
        mot_buffer = [qeye(mode.N) for mode in self.selected_modes]

        dims = self.selected_modes[mode_indx].N
        mot_buffer[mode_indx] = destroy(N=dims).dag() * destroy(N=dims)

        return QuantumOperator(qobj=tensor(*ion_buffer, *mot_buffer), name=name)


class QuantumOperator:
    """Class for representing quantum operators of which to take expectation values of

    Args:
        qobj (Qobj): Qutip-compatible object
        name (str): alias/name to reference the operator by
    """

    def __init__(self, qobj, name):
        self.qobj = qobj
        self.name = name
