# Copyright 2024-2025 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools as itr
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import sympy

from oqd_trical.misc import constants as cst

########################################################################################
from oqd_trical.misc.polynomial import polyder, polyval1d, polyval2d, polyval3d

########################################################################################
from .base import Base


@jax.jit
def _coulomb_potential(x, pair_i, pair_j, q):
    nxij = jnp.linalg.norm(x[pair_i] - x[pair_j], axis=-1)
    return cst.k_e * q**2 * jnp.sum(1 / nxij)


@partial(jax.jit, static_argnames=["axis", "ion"])
def _coulomb_first_derivative(x, other_indices, axis, ion, q):
    xia = x[ion, axis]
    xja = x[other_indices, axis]
    nxij = jnp.linalg.norm(x[ion] - x[other_indices], axis=-1)
    return cst.k_e * q**2 * jnp.sum((xja - xia) / nxij**3)


@partial(
    jax.jit,
    static_argnames=["axis_a", "axis_b", "ion_i", "ion_j"],
)
def _coulomb_second_derivative(
    x, other_indices, axis_a, axis_b, ion_i, ion_j, q
):
    if ion_i == ion_j:
        xia = x[ion_i, axis_a]
        xka = x[other_indices, axis_a]
        xib = x[ion_i, axis_b]
        xkb = x[other_indices, axis_b]
        nxik = jnp.linalg.norm(x[ion_i] - x[other_indices], axis=-1)
        if axis_a == axis_b:
            return cst.k_e * q**2 * jnp.sum(
                -1 / nxik**3 + 3 * (xka - xia) ** 2 / nxik**5
            )
        return cst.k_e * q**2 * jnp.sum(
            3 * (xka - xia) * (xkb - xib) / nxik**5
        )

    xia = x[ion_i, axis_a]
    xja = x[ion_j, axis_a]
    xib = x[ion_i, axis_b]
    xjb = x[ion_j, axis_b]
    nxij = jnp.linalg.norm(x[ion_i] - x[ion_j])
    if axis_a == axis_b:
        return cst.k_e * q**2 * (
            1 / nxij**3 - 3 * (xja - xia) ** 2 / nxij**5
        )
    return cst.k_e * q**2 * (-3 * (xja - xia) * (xjb - xib) / nxij**5)


@partial(jax.jit, static_argnames=["dim"])
def _polynomial_potential(x, alpha, dim):
    functions = {1: polyval1d, 2: polyval2d, 3: polyval3d}
    return jnp.sum(functions[dim](*jnp.asarray(x).transpose(), alpha))


@partial(jax.jit, static_argnames=["dim", "ion"])
def _polynomial_derivative(x, coefficients, dim, ion):
    functions = {1: polyval1d, 2: polyval2d, 3: polyval3d}
    return functions[dim](*jnp.asarray(x)[ion], coefficients)


@jax.jit
def _gaussian_optical_potential(x, focal_point, beam_waist, x_R, V):
    delta_x = x - focal_point
    w = beam_waist * jnp.sqrt(1 + (delta_x[:, 0] / x_R) ** 2)
    r = jnp.sqrt(delta_x[:, 1] ** 2 + delta_x[:, 2] ** 2 + 1e-5)
    e = jnp.exp(-2 * r**2 / w**2)
    return jnp.sum(V * e * beam_waist**2 / w**2)


@partial(jax.jit, static_argnames=["axis", "ion"])
def _gaussian_first_derivative(x, focal_point, beam_waist, x_R, V, axis, ion):
    delta_x = x[ion] - focal_point
    w = beam_waist * jnp.sqrt(1 + (delta_x[0] / x_R) ** 2)
    r = jnp.sqrt(delta_x[1] ** 2 + delta_x[2] ** 2)
    e = jnp.exp(-2 * r**2 / w**2)
    if axis == 0:
        return (2 * V * e * beam_waist**4 * delta_x[0] * (2 * r**2 - w**2)) / (
            w**6 * x_R**2
        )
    return -4 * V * e * beam_waist**2 * delta_x[axis] / w**4


@partial(
    jax.jit,
    static_argnames=["axis_a", "axis_b", "ion_i", "ion_j"],
)
def _gaussian_second_derivative(
    x, focal_point, beam_waist, x_R, V, axis_a, axis_b, ion_i, ion_j
):
    if ion_i != ion_j:
        return jnp.asarray(0.0)

    delta_x = x[ion_i] - focal_point
    w = beam_waist * jnp.sqrt(1 + (delta_x[0] / x_R) ** 2)
    r = jnp.sqrt(delta_x[1] ** 2 + delta_x[2] ** 2)
    e = jnp.exp(-2 * r**2 / w**2)

    if axis_a == axis_b == 0:
        return (
            -2
            * V
            * beam_waist**4
            * (
                w**6 * x_R**2
                - 4 * w**4 * beam_waist**2 * delta_x[0] ** 2
                + 8 * w**2 * beam_waist**2 * delta_x[0] ** 2 * r**2
                - 2
                * r**2
                * (
                    w**4 * x_R**2
                    - 4 * w**2 * beam_waist**2 * delta_x[0] ** 2
                    + 4 * beam_waist**2 * delta_x[0] ** 2 * r**2
                )
            )
            * e
            / (w**10 * x_R**4)
        )
    if axis_a == axis_b:
        return (
            -4
            * V
            * beam_waist**2
            * (w**2 - 4 * delta_x[axis_a] ** 2)
            * e
            / w**6
        )
    if axis_a == 0:
        return (
            16
            * V
            * beam_waist**4
            * delta_x[0]
            * delta_x[axis_b]
            * (w**2 - r**2)
            * e
            / (w**8 * x_R**2)
        )
    if axis_b == 0:
        return (
            16
            * V
            * beam_waist**4
            * delta_x[0]
            * delta_x[axis_a]
            * (w**2 - r**2)
            * e
            / (w**8 * x_R**2)
        )
    return 16 * V * beam_waist**2 * delta_x[1] * delta_x[2] * e / w**6


class Potential(Base):
    """
    Object representing a general potential.

    Args:
        d2phi (Callable): Function that takes two strings representing the derivative variables and outputs the function corresponding to the derivative of the potential with respect to the derivative variables.
        dphi (Callable): Function that takes a string representing the derivative variable and outputs the function corresponding to the derivative of the potential with respect to the derivative variable.
        phi (Callable ): Function representing the potential.

    Keyword Args:
        dim (int): Dimension of system.
    """

    def __init__(self, phi, dphi, d2phi, **kwargs):
        super(Potential, self).__init__()

        self.phi = phi
        self.dphi = dphi
        self.d2phi = d2phi

        params = {"dim": 3}
        params.update(kwargs)
        self.__dict__.update(params)
        self.params = params
        pass

    def __add__(self, other):
        for i in np.intersect1d(list(self.params.keys()), list(other.params.keys())):
            assert self.params[i] == other.params[i], (
                "Potentials with incompatible dimensions"
            )

        params = {}
        params.update(self.params)
        params.update(other.params)

        def phi(x):
            return self.phi(x) + other.phi(x)

        def dphi(var):
            return lambda x: self.dphi(var)(x) + other.dphi(var)(x)

        def d2phi(var1, var2):
            return lambda x: self.d2phi(var1, var2)(x) + other.d2phi(var1, var2)(x)

        return Potential(phi, dphi, d2phi, **params)

    def __sub__(self, other):
        for i in np.intersect1d(list(self.params.keys()), list(other.params.keys())):
            assert self.params[i] == other.params[i], (
                "Potentials with incompatible dimensions"
            )

        params = {}
        params.update(self.params)
        params.update(other.params)

        def phi(x):
            return self.phi(x) - other.phi(x)

        def dphi(var):
            return lambda x: self.dphi(var)(x) - other.dphi(var)(x)

        def d2phi(var1, var2):
            return lambda x: self.d2phi(var1, var2)(x) - other.d2phi(var1, var2)(x)

        return Potential(phi, dphi, d2phi, **params)

    def __mul__(self, multiplier):
        def phi(x):
            return self.phi(x) * multiplier

        def dphi(var):
            return lambda x: self.dphi(var)(x) * multiplier

        def d2phi(var1, var2):
            return lambda x: self.d2phi(var1, var2)(x) * multiplier

        return Potential(phi, dphi, d2phi, **self.params)

    def __rmul__(self, multiplier):
        return self * multiplier

    def __truediv__(self, divisor):
        def phi(x):
            return self.phi(x) / divisor

        def dphi(var):
            return lambda x: self.dphi(var)(x) / divisor

        def d2phi(var1, var2):
            return lambda x: self.d2phi(var1, var2)(x) / divisor

        return Potential(phi, dphi, d2phi, **self.params)

    def __call__(self, x):
        return self.phi(x)

    def first_derivative(self, var):
        """
        Calculates the first derivative of the potential with respect to a variable.

        Args:
            var (str): Derivative variable.

        Returns:
            (Callable): Function corresponding to the first derivative of the potential with respect to the derivative variable.
        """
        return self.dphi(var)

    def second_derivative(self, var1, var2):
        """
        Calculates the second derivative of the potential with respect to two variables.

        Args:
            var1 (str): first derivative variable.
            var2 (str): second derivative variable.

        Returns:
            (Callable): Function corresponding to the second derivative of the potential with respect to the derivative variables.
        """
        return self.d2phi(var1, var2)

    def gradient(self):
        """
        Calculates the gradient of the potential.

        Returns:
            (Callable): Function corresponding to the gradient of the potential.
        """

        def grad_phi(x):
            return jnp.asarray(
                [
                    self.dphi(var)(x)
                    for var in itr.product(
                        ["x", "y", "z"][: self.dim], np.arange(self.N, dtype=int)
                    )
                ]
            )

        return jax.jit(grad_phi)

    def hessian(self):
        """
        Calculates the Hessian of the potential.

        Returns:
            (Callable): Function corresponding to the Hessian of the potential.
        """

        def hess_phi(x):
            variables = list(
                itr.product(["x", "y", "z"][: self.dim], np.arange(self.N, dtype=int))
            )
            return jnp.asarray(
                [[self.d2phi(var1, var2)(x) for var2 in variables] for var1 in variables]
            )

        return jax.jit(hess_phi)

    def nondimensionalize(self, l):  # noqa: E741
        """
        Nondimensionalizes a Potential with a length scale.

        Args:
            l (float): Length scale.

        Returns:
            (Potential): Potential representing the nondimensionalized coulomb potential.
        """

        def nd_phi(x):
            return self.phi(x * l)

        def nd_dphi(var):
            return lambda x: self.dphi(var)(x * l)

        def nd_d2phi(var1, var2):
            return lambda x: self.d2phi(var1, var2)(x * l)

        return (
            Potential(nd_phi, nd_dphi, nd_d2phi, **self.params)
            * l
            / (cst.k_e * cst.e**2)
        )

    def update_params(self, **kwargs):
        """
        Updates parameters, i.e. params attribute, of a Potential object.

        Args:
            dim (int): Dimension of the system.
            N (int): Number of Ions.
        """
        self.params.update(kwargs)
        self.__dict__.update(self.params)
        pass

    pass


########################################################################################


class CoulombPotential(Potential):
    """
    Object representing a coulomb potential.

    Args:
        N (int): Number of ions.

    Keyword Args:
        dim (int): Dimension of system.
        N (int): Number of ions.
        q (float): Charge of ions.
    """

    def __init__(self, N, **kwargs):
        params = {"dim": 3, "N": N, "q": cst.e}
        params.update(kwargs)
        pair_indices = (
            np.fromiter(itr.chain(*itr.combinations(range(N), 2)), dtype=int)
            .reshape(-1, 2)
            .transpose()
        )
        self.pair_i = pair_indices[0]
        self.pair_j = pair_indices[1]

        super(CoulombPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        return _coulomb_potential(x, self.pair_i, self.pair_j, self.q)

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])
        j = np.delete(np.arange(self.N, dtype=int), i)

        def dphi_dai(x):
            return _coulomb_first_derivative(x, j, a, i, self.q)

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])
        k = np.delete(np.arange(self.N, dtype=int), i)

        def d2phi_daidbj(x):
            return _coulomb_second_derivative(x, k, a, b, i, j, self.q)

        return d2phi_daidbj

    def nondimensionalize(self, l):  # noqa: E741
        return self / (cst.k_e * cst.e**2)

    pass


class PolynomialPotential(Potential):
    """
    Object representing a polynomial potential.

    Args:
        alpha (np.ndarray[float]): Coefficients of the polynomial potential.

    Keyword Args:
        dim (int): Dimension of system.
    """

    def __init__(self, alpha, **kwargs):
        self.alpha = np.array(alpha)
        self.deg = np.array(alpha.shape)

        params = {"dim": len(alpha.shape)}
        params.update(kwargs)

        super(PolynomialPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        return _polynomial_potential(x, self.alpha, self.dim)

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])

        beta = polyder(self.alpha, axis=a)

        def dphi_dai(x):
            return _polynomial_derivative(x, beta, self.dim, i)

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])

        beta = polyder(self.alpha, axis=a)
        gamma = polyder(beta, axis=b)

        if i == j:

            def d2phi_daidbj(x):
                return _polynomial_derivative(x, gamma, self.dim, i)
        else:

            def d2phi_daidbj(x):
                return 0.0

        return d2phi_daidbj

    def nondimensionalize(self, l):  # noqa: E741
        alpha = (
            l ** np.indices(self.alpha.shape).sum(0)
            * self.alpha
            * (l / (cst.k_e * cst.e**2))
        )
        return PolynomialPotential(alpha, **self.params)

    pass


class GaussianOpticalPotential(Potential):
    """
    Object representing a potential caused by a Gaussian beam.

    Args:
        focal_point (np.ndarray[float]): Center of the Gaussian beam.
        power (float): Power of Gaussian beam.
        wavelength (float): Wavelength of Gaussian beam.
        beam_waist (float): Waist of Gaussian beam.

    Keyword Args:
        dim (int): Dimension of system.
        m (float): Mass of ions.
        Omega_bar (float): Rabi frequency per root intensity.
        transition_wavelength (float): Wavelength of the transition that creates the optical trap.
        refractive_index (float): Refractive index of medium Gaussian beam is propagating through.
    """

    def __init__(self, focal_point, power, wavelength, beam_waist, **opt_kwargs):
        self.params = {"dim": 3}

        opt_params = {
            "m": 171 * cst.m_u,
            "Omega_bar": 2.23e6,
            "transition_wavelength": 369.52e-9,
            "refractive_index": 1,
            "focal_point": focal_point,
            "power": power,
            "wavelength": wavelength,
            "beam_waist": beam_waist,
        }
        opt_params.update(opt_kwargs)
        self.__dict__.update(opt_params)
        self.opt_params = opt_params

        nu = cst.convert_lamb_to_omega(wavelength)
        nu_transition = cst.convert_lamb_to_omega(opt_params["transition_wavelength"])
        Delta = nu - nu_transition
        x_R = np.pi * beam_waist**2 * opt_params["refractive_index"] / wavelength
        I = 2 * power / (np.pi * beam_waist**2)  # noqa: E741
        Omega = opt_params["Omega_bar"] * np.sqrt(np.abs(I))
        omega_x = np.sqrt(
            np.abs(
                cst.hbar
                * self.Omega_bar**2
                * power
                * wavelength**2
                / (self.refractive_index**2 * np.pi**3 * Delta * beam_waist**6 * self.m)
            )
        )
        omega_y = omega_z = np.sqrt(
            np.abs(
                2
                * cst.hbar
                * self.Omega_bar**2
                * power
                / (np.pi * Delta * beam_waist**4 * self.m)
            )
        )

        self.nu = nu
        self.nu_transition = nu_transition
        self.Delta = Delta
        self.x_R = x_R
        self.I = I
        self.Omega = Omega
        self.stark_shift = np.abs(Omega**2 / (4 * Delta))
        self.V = cst.hbar * self.Omega_bar**2 * self.I / (4 * self.Delta)
        self.omega = np.array([omega_x, omega_y, omega_z])

        super(GaussianOpticalPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **self.params
        )
        pass

    def __call__(self, x):
        return _gaussian_optical_potential(
            x, self.focal_point, self.beam_waist, self.x_R, self.V
        )

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])

        def dphi_dai(x):
            return _gaussian_first_derivative(
                x, self.focal_point, self.beam_waist, self.x_R, self.V, a, i
            )

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])

        def d2phi_daidbj(x):
            return _gaussian_second_derivative(
                x, self.focal_point, self.beam_waist, self.x_R, self.V, a, b, i, j
            )

        return d2phi_daidbj

    def nondimensionalize(self, l):  # noqa: E741
        ndgop = (
            GaussianOpticalPotential(
                self.focal_point / l,
                self.power,
                self.wavelength,
                self.beam_waist / l,
                m=self.m,
                Omega_bar=self.Omega_bar / l,
                transition_wavelength=self.transition_wavelength,
                refractive_index=self.refractive_index,
            )
            * l
            / (cst.k_e * cst.e**2)
        )
        ndgop.update_params(**self.params)
        return ndgop

    pass


########################################################################################


class SymbolicPotential(Potential):
    """
    Object representing a symbolically defined potential, same for all ions.

    Args:
        expr (str): Symbolic expression of the potential.

    Keyword Args:
        dim (int): Dimension of system.
    """

    def __init__(self, expr, **kwargs):
        self.expr = expr

        params = {"dim": 3}
        params.update(kwargs)
        self.__dict__.update(params)
        self.params = params

        self.symbol = [sympy.Symbol(["x", "y", "z"][i]) for i in range(self.dim)]
        self.lambdified_expr = jax.jit(
            sympy.utilities.lambdify(self.symbol, expr, "jax")
        )

        super(SymbolicPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        return jnp.sum(self.lambdified_expr(*jnp.asarray(x).transpose()))

    def evaluate(self, x):
        return self.lambdified_expr(*jnp.asarray(x).transpose())

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])
        derivative = jax.jit(
            sympy.utilities.lambdify(
                self.symbol, sympy.diff(self.expr, self.symbol[a]), "jax"
            )
        )

        def dphi_dai(x):
            return derivative(*jnp.asarray(x)[i])

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])

        if i == j:
            derivative = jax.jit(
                sympy.utilities.lambdify(
                    self.symbol,
                    sympy.diff(self.expr, self.symbol[a], self.symbol[b]),
                    "jax",
                )
            )

            def d2phi_daidbj(x):
                return derivative(*jnp.asarray(x)[i])
        else:

            def d2phi_daidbj(x):
                return 0

        return d2phi_daidbj

    def nondimensionalize(self, l):  # noqa: E741
        expr = self.expr.subs({k: k * l for k in self.symbol}) * (
            l / (cst.k_e * cst.e**2)
        )
        return SymbolicPotential(expr, **self.params)

    pass


class AdvancedSymbolicPotential(Potential):
    """
    Object representing a symbolically defined potential that need not be the same for all ions.

    Args:
        expr (str): Symbolic expression of the potential.

    Keyword Args:
        dim (int): Dimension of system.
        N (int): Number of ions.
    """

    def __init__(self, N, expr, **kwargs):
        self.expr = expr

        params = {"dim": 3, "N": N}
        params.update(kwargs)
        self.__dict__.update(params)
        self.params = params

        self.symbol = np.array(
            [
                [
                    sympy.Symbol(["x{}", "y{}", "z{}"][i].format(j))
                    for i in range(self.dim)
                ]
                for j in range(N)
            ]
        ).flatten()
        self.lambdified_expr = jax.jit(
            sympy.utilities.lambdify(self.symbol, expr, "jax")
        )

        super(AdvancedSymbolicPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        x = jnp.asarray(x)
        return self.lambdified_expr(*x.flatten())

    def first_derivative(self, var):
        a = var[0]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])
        axis = {"x": 0, "y": 1, "z": 2}[a]
        derivative = jax.jit(
            sympy.utilities.lambdify(
                self.symbol, sympy.diff(self.expr, self.symbol[i * self.dim + axis]), "jax"
            )
        )

        def dphi_dai(x):
            return derivative(*jnp.asarray(x).flatten())

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = var1[0]
        b = var2[0]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])
        axis_a = {"x": 0, "y": 1, "z": 2}[a]
        axis_b = {"x": 0, "y": 1, "z": 2}[b]
        derivative = jax.jit(
            sympy.utilities.lambdify(
                self.symbol,
                sympy.diff(
                    self.expr,
                    self.symbol[i * self.dim + axis_a],
                    self.symbol[j * self.dim + axis_b],
                ),
                "jax",
            )
        )

        def d2phi_daidbj(x):
            return derivative(*jnp.asarray(x).flatten())

        return d2phi_daidbj

    def nondimensionalize(self, l):  # noqa: E741
        expr = self.expr.subs({k: k * l for k in self.symbol}) * (
            l / (cst.k_e * cst.e**2)
        )
        params = self.params
        if "N" in params.keys():
            params.pop("N")
        return AdvancedSymbolicPotential(self.N, expr, **params)

    pass


class SymbolicOpticalPotential(SymbolicPotential):
    """
    Object representing a general optical potential symbolically.

    Args:
        intensity_expr (str): Expression for the intensity of the optical potential.
        wavelength (float): Wavelength of the optical potential.

    Keyword Args:
        dim (int): Dimension of system.
        m (float): Mass of ions.
        Omega_bar (float): Rabi frequency per root intensity.
        transition_wavelength (float): Wavelength of the transition that creates the optical trap.
        refractive_index (float): Refractive index of medium Gaussian beam is propagating through.
    """

    def __init__(self, intensity_expr, wavelength, **kwargs):
        self.params = {"dim": 3}

        self.intensity_expr = intensity_expr
        self.wavelength = wavelength

        opt_params = {
            "m": 171 * cst.m_u,
            "Omega_bar": 2.23e6,
            "transition_wavelength": 369.52e-9,
            "refractive_index": 1,
        }
        opt_params.update(kwargs)
        self.__dict__.update(opt_params)
        self.opt_params = opt_params

        nu = cst.convert_lamb_to_omega(wavelength)
        nu_transition = cst.convert_lamb_to_omega(opt_params["transition_wavelength"])
        Delta = nu - nu_transition

        self.nu = nu
        self.nu_transition = nu_transition
        self.Delta = Delta

        expr = cst.hbar * opt_params["Omega_bar"] ** 2 * intensity_expr / (4 * Delta)

        super(SymbolicOpticalPotential, self).__init__(expr, **self.params)
        pass

    pass


########################################################################################

########################################################################################


class AutoDiffPotential(Potential):
    """
    Object representing a functionally defined potential for the system of ions that uses automatic differentiation to calculate derivatives of the potential.

    Args:
        expr (Callable): function of the potential that is defined using jax.numpy.

    Keyword Args:
        dim (int): Dimension of system.
    """

    def __init__(self, expr, **kwargs):
        self.expr = expr

        params = {"dim": 3}
        params.update(kwargs)
        self.__dict__.update(params)
        self.params = params

        super(AutoDiffPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        return self.expr(x)

    def gradient(self):
        def flatten_expr(x):
            return self.expr(x.reshape(self.dim, -1).transpose())

        jacobian = jax.jit(jax.jacobian(flatten_expr, 0))

        @jax.jit
        def grad(x):
            return jacobian(jnp.asarray(x).transpose().reshape(-1))

        return grad

    def hessian(self):
        def flatten_expr(x):
            return self.expr(x.reshape(self.dim, -1).transpose())

        hessian = jax.jit(jax.hessian(flatten_expr, 0))

        @jax.jit
        def hess(x):
            return hessian(jnp.asarray(x).transpose().reshape(-1))

        return hess

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])
        return lambda x: self.gradient()(x)[a * self.N + i]

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])
        return lambda x: self.hessian()(x)[a * self.N + i][b * self.N + j]

    def nondimensionalize(self, l):  # noqa: E741
        def expr(x):
            return self.expr(x * l) * l / (cst.k_e * cst.e**2)

        ndadp = AutoDiffPotential(expr, **self.params)
        ndadp.update_params(**self.params)
        return ndadp

    pass


class OpticalPotential(AutoDiffPotential):
    """
    Object representing a general optical potential functionally using automatic differentiation to calculate the derivatives.

    Args:
        intensity_expr (Callable): function of the expression for intensity of the optical potential that is defined using jax.numpy.
        wavelength (float): Wavelength of the optical potential.

    Keyword Args:
        dim (int): Dimension of system.
        m (float): Mass of ions.
        Omega_bar (float): Rabi frequency per root intensity.
        transition_wavelength (float): Wavelength of the transition that creates the optical trap.
        refractive_index (float): Refractive index of medium Gaussian beam is propagating through.
    """

    def __init__(self, intensity_expr, wavelength, **opt_kwargs):
        self.params = {"dim": 3}

        self.intensity_expr = intensity_expr
        self.wavelength = wavelength

        opt_params = {
            "m": 171 * cst.m_u,
            "Omega_bar": 2.23e6,
            "transition_wavelength": 369.52e-9,
            "refractive_index": 1,
            "wavelength": wavelength,
        }
        opt_params.update(opt_kwargs)
        self.__dict__.update(opt_params)
        self.opt_params = opt_params

        nu = cst.convert_lamb_to_omega(wavelength)
        nu_transition = cst.convert_lamb_to_omega(opt_params["transition_wavelength"])
        Delta = nu - nu_transition

        self.nu = nu
        self.nu_transition = nu_transition
        self.Delta = Delta

        def expr(x):
            return (
                cst.hbar
                * opt_params["Omega_bar"] ** 2
                * intensity_expr(x)
                / (4 * Delta)
            )

        super(OpticalPotential, self).__init__(expr, **self.params)
        pass

    pass
