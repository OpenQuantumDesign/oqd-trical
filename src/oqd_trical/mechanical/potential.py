from functools import partial
from numbers import Number

import jax
from jax import numpy as jnp

########################################################################################


class Potential:
    def __init__(self, phi):
        self.phi = jax.jit(phi)
        pass

    def __call__(self, x):
        x = jnp.array(x, dtype=jnp.float32)
        return self.phi(x)

    def grad(self, x):
        x = jnp.array(x, dtype=jnp.float32)
        return jax.grad(self.phi)(x)

    def hessian(self, x):
        x = jnp.array(x, dtype=jnp.float32)
        return jax.hessian(self.phi)(x)

    def __add__(self, other):
        assert isinstance(other, Potential)
        return PotentialAdd(self, other)

    def __mul__(self, other):
        assert isinstance(other, Number)
        return PotentialScalarMul(self, other)

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        assert isinstance(other, Number)
        return PotentialAdd(self, -1 * other)

    def __div__(self, other):
        assert isinstance(other, Number)
        return PotentialScalarMul(self, 1 / other)


########################################################################################


class PotentialAdd(Potential):
    def __init__(self, pot1, pot2):
        self.pot1 = pot1
        self.pot2 = pot2

        def phi(x):
            return self.pot1(x) + self.pot2(x)

        super().__init__(phi)


class PotentialScalarMul(Potential):
    def __init__(self, pot, multiplier):
        self.pot = pot
        self.multiplier = multiplier

        def phi(x):
            return self.pot(x) * self.multiplier

        super().__init__(phi)


########################################################################################


class CoulombPotential(Potential):
    def __init__(self):
        def phi(x):
            idcs = jnp.triu_indices(n=x.shape[-1], k=1)

            D = jnp.linalg.norm(x[:, idcs[0]] - x[:, idcs[1]], axis=0)

            return D.sum()

        super().__init__(phi)


########################################################################################


class PolynomialPotential(Potential):
    def __init__(self, alpha):
        self.alpha = alpha
        phi = partial(PolynomialPotential.polyval, alpha=alpha)
        super().__init__(phi)

    @staticmethod
    def polyval(x, alpha):
        exponent = jnp.array(
            jnp.meshgrid(
                *[jnp.arange(s, dtype=jnp.int32) for s in alpha.shape], indexing="ij"
            )
        )

        return (
            (x[:, None, None, None, :] ** exponent[..., None]).prod(0)
            * alpha[..., None]
        ).sum()


class HarmonicPotential(PolynomialPotential):
    def __init__(self, omega):
        self.omega = omega

        alpha = jnp.zeros((3, 3, 3))
        alpha = alpha.at[tuple(jnp.eye(3, dtype=int) * 2)].set(jnp.array([5, 5, 1]))

        super().__init__(alpha)


########################################################################################


class OpticalPotential(Potential):
    def __init__(self):
        raise NotImplementedError


class GaussianOpticalPotential(OpticalPotential):
    def __init__(self):
        raise NotImplementedError


# class GaussianOpticalPotential(Potential):
#     """
#     Object representing a potential caused by a Gaussian beam.

#     Args:
#         focal_point (np.ndarray[float]): Center of the Gaussian beam.
#         power (float): Power of Gaussian beam.
#         wavelength (float): Wavelength of Gaussian beam.
#         beam_waist (float): Waist of Gaussian beam.

#     Keyword Args:
#         dim (int): Dimension of system.
#         m (float): Mass of ions.
#         Omega_bar (float): Rabi frequency per root intensity.
#         transition_wavelength (float): Wavelength of the transition that creates the optical trap.
#         refractive_index (float): Refractive index of medium Gaussian beam is propagating through.
#     """

#     def __init__(self, focal_point, power, wavelength, beam_waist, **opt_kwargs):
#         self.params = {"dim": 3}

#         opt_params = {
#             "m": cst.convert_m_a(171),
#             "Omega_bar": 2.23e6,
#             "transition_wavelength": 369.52e-9,
#             "refractive_index": 1,
#             "focal_point": focal_point,
#             "power": power,
#             "wavelength": wavelength,
#             "beam_waist": beam_waist,
#         }
#         opt_params.update(opt_kwargs)
#         self.__dict__.update(opt_params)
#         self.opt_params = opt_params

#         nu = cst.convert_lamb_to_omega(wavelength)
#         nu_transition = cst.convert_lamb_to_omega(opt_params["transition_wavelength"])
#         Delta = nu - nu_transition
#         x_R = np.pi * beam_waist**2 * opt_params["refractive_index"] / wavelength
#         I = 2 * power / (np.pi * beam_waist**2)
#         Omega = opt_params["Omega_bar"] * np.sqrt(np.abs(I))
#         omega_x = np.sqrt(
#             np.abs(
#                 cst.hbar
#                 * self.Omega_bar**2
#                 * power
#                 * wavelength**2
#                 / (self.refractive_index**2 * np.pi**3 * Delta * beam_waist**6 * self.m)
#             )
#         )
#         omega_y = omega_z = np.sqrt(
#             np.abs(
#                 2
#                 * cst.hbar
#                 * self.Omega_bar**2
#                 * power
#                 / (np.pi * Delta * beam_waist**4 * self.m)
#             )
#         )

#         self.nu = nu
#         self.nu_transition = nu_transition
#         self.Delta = Delta
#         self.x_R = x_R
#         self.I = I
#         self.Omega = Omega
#         self.stark_shift = np.abs(Omega**2 / (4 * Delta))
#         self.V = cst.hbar * self.Omega_bar**2 * self.I / (4 * self.Delta)
#         self.omega = np.array([omega_x, omega_y, omega_z])

#         super(GaussianOpticalPotential, self).__init__(
#             self.__call__, self.first_derivative, self.second_derivative, **self.params
#         )
#         pass

#     def __call__(self, x):
#         delta_x = x - self.focal_point
#         w0 = self.beam_waist
#         w = w0 * np.sqrt(1 + (delta_x[:, 0] / self.x_R) ** 2)
#         V = self.V
#         r = np.sqrt(delta_x[:, 1] ** 2 + delta_x[:, 2] ** 2)
#         e = np.exp(-2 * r**2 / w**2)
#         return (V * e * w0**2 / w**2).sum()

#     def first_derivative(self, var):
#         a = {"x": 0, "y": 1, "z": 2}[var[0]]
#         i = int(var[1:] if isinstance(var, str) else var[1:][0])

#         def dphi_dai(x):
#             V = self.V
#             w0 = self.beam_waist
#             xR = self.x_R
#             delta_x = x[i] - self.focal_point
#             w = w0 * np.sqrt(1 + (delta_x[0] / xR) ** 2)
#             r = np.sqrt(delta_x[1] ** 2 + delta_x[2] ** 2)
#             e = np.exp(-2 * r**2 / w**2)
#             if a == 0:
#                 return (2 * V * e * w0**4 * delta_x[0] * (2 * r**2 - w**2)) / (
#                     w**6 * xR**2
#                 )
#             else:
#                 return -4 * V * e * w0**2 * delta_x[a] / w**4

#         return dphi_dai

#     def second_derivative(self, var1, var2):
#         a = {"x": 0, "y": 1, "z": 2}[var1[0]]
#         b = {"x": 0, "y": 1, "z": 2}[var2[0]]
#         i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
#         j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])

#         def d2phi_daidbj(x):
#             V = self.V
#             w0 = self.beam_waist
#             xR = self.x_R
#             delta_x = x[i] - self.focal_point
#             w = w0 * np.sqrt(1 + (delta_x[0] / xR) ** 2)
#             r = np.sqrt(delta_x[1] ** 2 + delta_x[2] ** 2)
#             e = np.exp(-2 * r**2 / w**2)
#             if i != j:
#                 return 0
#             else:
#                 if a == b == 0:
#                     return (
#                         -2
#                         * V
#                         * w0**4
#                         * (
#                             w**6 * xR**2
#                             - 4 * w**4 * w0**2 * delta_x[0] ** 2
#                             + 8 * w**2 * w0**2 * delta_x[0] ** 2 * r**2
#                             - 2
#                             * r**2
#                             * (
#                                 w**4 * xR**2
#                                 - 4 * w**2 * w0**2 * delta_x[0] ** 2
#                                 + 4 * w0**2 * delta_x[0] ** 2 * r**2
#                             )
#                         )
#                         * e
#                         / (w**10 * xR**4)
#                     )
#                 elif a == b:
#                     return -4 * V * w0**2 * (w**2 - 4 * delta_x[a] ** 2) * e / w**6

#                 elif a == 0:
#                     return (
#                         16
#                         * V
#                         * w0**4
#                         * delta_x[0]
#                         * delta_x[b]
#                         * (w**2 - r**2)
#                         * e
#                         / (w**8 * xR**2)
#                     )
#                 elif b == 0:
#                     return (
#                         16
#                         * V
#                         * w0**4
#                         * delta_x[0]
#                         * delta_x[a]
#                         * (w**2 - r**2)
#                         * e
#                         / (w**8 * xR**2)
#                     )
#                 else:
#                     return 16 * V * w0**2 * delta_x[1] * delta_x[2] * e / w**6

#         return d2phi_daidbj

#     def nondimensionalize(self, l):
#         ndgop = (
#             GaussianOpticalPotential(
#                 self.focal_point / l,
#                 self.power,
#                 self.wavelength,
#                 self.beam_waist / l,
#                 m=self.m,
#                 Omega_bar=self.Omega_bar / l,
#                 transition_wavelength=self.transition_wavelength,
#                 refractive_index=self.refractive_index,
#             )
#             * l
#             / (cst.k_e * cst.e**2)
#         )
#         ndgop.update_params(**self.params)
#         return ndgop

#     pass


# ########################################################################################


# class OpticalPotential(AutoDiffPotential):
#     """
#     Object representing a general optical potential functionally using automatic differentiation to calculate the derivatives.

#     Args:
#         intensity_expr (Callable): function of the expression for intensity of the optical potential that is defined using the numpy submodule of autograd package.
#         wavelength (float): Wavelength of the optical potential.

#     Keyword Args:
#         dim (int): Dimension of system.
#         m (float): Mass of ions.
#         Omega_bar (float): Rabi frequency per root intensity.
#         transition_wavelength (float): Wavelength of the transition that creates the optical trap.
#         refractive_index (float): Refractive index of medium Gaussian beam is propagating through.
#     """

#     def __init__(self, intensity_expr, wavelength, **opt_kwargs):
#         self.params = {"dim": 3}

#         self.intensity_expr = intensity_expr
#         self.wavelength = wavelength

#         opt_params = {
#             "m": cst.convert_m_a(171),
#             "Omega_bar": 2.23e6,
#             "transition_wavelength": 369.52e-9,
#             "refractive_index": 1,
#             "wavelength": wavelength,
#         }
#         opt_params.update(opt_kwargs)
#         self.__dict__.update(opt_params)
#         self.opt_params = opt_params

#         nu = cst.convert_lamb_to_omega(wavelength)
#         nu_transition = cst.convert_lamb_to_omega(opt_params["transition_wavelength"])
#         Delta = nu - nu_transition

#         self.nu = nu
#         self.nu_transition = nu_transition
#         self.Delta = Delta

#         def expr(x):
#             return (
#                 cst.hbar
#                 * opt_params["Omega_bar"] ** 2
#                 * intensity_expr(x)
#                 / (4 * Delta)
#             )

#         super(OpticalPotential, self).__init__(expr, **self.params)
#         pass

#     pass
