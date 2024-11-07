# from numbers import Number
# from math import factorial, sqrt, exp
# from typing import List, Union, Callable, Optional
# import numpy as np
# from scipy.special import genlaguerre as L
# from qutip import basis, Qobj, tensor, qeye, destroy
# from pydantic import BaseModel, validator, Field

# ########################################################################################

# from ...classes import PolynomialPotential, TrappedIons

# ########################################################################################

# # class VibrationalMode(BaseModel):
# #     """
# #     Class representing vibrational mode of an ion chain.

# #     Attributes:
# #         eigenfreq (float): Eigenfrequency in Hz of the mode.
# #         eigenvect (List[float]): Eigenvector of the mode.
# #         axis (List[float]): Mode axis of oscillation.
# #         N (int): Max phonon cutoff.
# #     """

# #     eigenfreq: float
# #     eigenvect: List[float]
#     # axis: List[float]
#     # N: int

#     # class Config:
#     #     arbitrary_types_allowed = True

# #MUST BE RULES CHANGE:
#     # def setstate(self, n: int) -> Qobj:
#     #     """Access the mode's nth excited state."""
#     #     if n >= self.N:
#     #         raise ValueError("Outside of Hilbert space")
#     #     return basis(self.N, n)

#     # def groundstate(self) -> Qobj:
#     #     """Access the mode's ground state."""
#     #     return basis(self.N, 0)

#     # def modecutoff(self, val: int):
#     #     """Set the mode cutoff."""
#     #     self.N = val + 1

# # class Laser(BaseModel):
# #     """
# #     Class representing a laser.

# #     Attributes:
# #         wavelength (float): Laser's wavelength.
# #         k_hat (List[float]): Laser's normalized wave vector.
# #         I (Union[Number, Callable]): Laser's intensity in W/m^2.
# #         eps_hat (List[float]): Laser's normalized polarization vector.
# #         phi (float): Laser's phase.
# #         detuning (float): How much to detune laser's frequency by in Hz.
# #     """

# #     wavelength: Optional[float] = None
# #     k_hat: Optional[List[float]] = None
# #     I: Union[Number, Callable[[float], float]]
# #     I_0: float = None
# #     eps_hat: List[float] = Field(default_factory=lambda: [0, 0, 1])
# #     phi: float = 0.0
# #     detuning: float = 0.0

# #     class Config:
# #         arbitrary_types_allowed = True

#     # @validator('I')
#     # def set_intensity_fn(cls, v):
#     #     if isinstance(v, Number):
#     #         # Set as constant function
#     #         return lambda t: v
#     #     elif callable(v):
#     #         return v
#     #     else:
#     #         raise ValueError("I must be a number or a callable function")

# #ASSIGNMENTS FOR LASER:

#     def detune(self, Delta: float):
#         """Set the laser's detuning."""
#         self.detuning = Delta

# # class Chain(BaseModel):
# #     """
# #     Class representing an ion chain.

# #     Attributes:
# #         ions (List): List of Ion objects.
# #         trap_freqs (List[float]): Trap COM frequencies in Hz [omega_x, omega_y, omega_z].
# #         selected_modes (List[VibrationalMode]): List of VibrationalMode objects.
# #         modes (List[VibrationalMode]): List of all VibrationalMode objects.
# #         eqm_pts (np.ndarray): Equilibrium positions of the ions.
# #     """
# #     def __init__(self, ions: List, trap_freqs: List[float], selected_modes: List[VibrationalMode], **kwargs):
# #         super().__init__(**kwargs)
# #         self.ions = ions
# #         self.trap_freqs = trap_freqs
# #         self.selected_modes = selected_modes
# #         self.modes = []
# #         self.eqm_pts = np.array([])
    
# #     ions: List
# #     trap_freqs: List[float]
# #     selected_modes: List[VibrationalMode]
# #     modes: List[VibrationalMode] = Field(default_factory=list)
# #     eqm_pts: np.ndarray = Field(default_factory=lambda: np.array([]))

#     class Config:
#         arbitrary_types_allowed = True

#     def __init__(self, **data):
#         super().__init__(**data)
#         N = len(self.ions)
#         mass = self.ions[0].mass

#         omega_x = 2 * np.pi * self.trap_freqs[0]
#         omega_y = 2 * np.pi * self.trap_freqs[1]
#         omega_z = 2 * np.pi * self.trap_freqs[2]

#         alpha = np.zeros((3, 3, 3))
#         alpha[2, 0, 0] = mass * (omega_x) ** 2 / 2
#         alpha[0, 2, 0] = mass * (omega_y) ** 2 / 2
#         alpha[0, 0, 2] = mass * (omega_z) ** 2 / 2

#         pp = PolynomialPotential(alpha, N=N)
#         ti = TrappedIons(N, pp, m=mass)
#         ti.principle_axis()

#         eigenfreqs = ti.w_pa / (2 * np.pi)
#         eigenvects = ti.b_pa

#         for l in range(len(eigenfreqs)):
#             if 0 <= l <= N - 1:
#                 axis = np.array([1, 0, 0])
#             elif N <= l <= 2 * N - 1:
#                 axis = np.array([0, 1, 0])
#             elif 2 * N <= l <= 3 * N - 1:
#                 axis = np.array([0, 0, 1])
#             else:
#                 raise ValueError("Frequency direction sorting went wrong :(")

#             self.modes.append(
#                 VibrationalMode(
#                     eigenfreq=eigenfreqs[l],
#                     eigenvect=eigenvects[l],
#                     axis=axis.tolist(),
#                     N=10
#                 )
#             )

#         self.eqm_pts = ti.equilibrium_position()

#     def ion_projector(self, ion_numbers: Union[int, List[int]], names: Union[str, List[str]]):
#         """Full Hilbert space projector onto internal state."""
#         mot_buffer = [qeye(mode.N) for mode in self.selected_modes]

#         if isinstance(names, str) and isinstance(ion_numbers, int):
#             # Single-ion case
#             ion_buffer = [qeye(ion.N_levels) for ion in self.ions]
#             ket = self.ions[ion_numbers - 1].state[names]
#             ion_buffer[ion_numbers - 1] = ket * ket.dag()
#             name = names
#         elif isinstance(names, list):
#             # Multi-ion case
#             ket = tensor(
#                 *[
#                     self.ions[ion_numbers[j] - 1].state[names[j]]
#                     for j in range(len(names))
#                 ]
#             )
#             ion_buffer = []
#             for j in range(len(self.ions)):
#                 if j + 1 not in ion_numbers:
#                     ion_buffer.append(qeye(self.ions[j].N_levels))
#                 elif j + 1 == ion_numbers[0]:
#                     ion_buffer.append(ket * ket.dag())
#             name = "".join(names)
#         else:
#             raise ValueError("Invalid types for ion_numbers or names")

#         return QuantumOperator(qobj=tensor(*ion_buffer, *mot_buffer), name=name)

#     def number_operator(self, mode_indx: int, name: str):
#         """Full Hilbert space number operator for a mode."""
#         ion_buffer = [qeye(ion.N_levels) for ion in self.ions]
#         mot_buffer = [qeye(mode.N) for mode in self.selected_modes]

#         dims = self.selected_modes[mode_indx].N
#         mot_buffer[mode_indx] = destroy(N=dims).dag() * destroy(N=dims)

#         return QuantumOperator(qobj=tensor(*ion_buffer, *mot_buffer), name=name)

# # class QuantumOperator(BaseModel):
# #     """
# #     Class representing quantum operators for expectation values.

# #     Attributes:
# #         qobj (Qobj): Qutip-compatible object.
# #         name (str): Alias/name for the operator.
# #     """

# #     qobj: Qobj
# #     name: str

# #     class Config:
# #         arbitrary_types_allowed = True
