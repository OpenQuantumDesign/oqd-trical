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

import autograd as ag
import numpy as np
import sympy
from numpy.polynomial import polynomial as poly

from oqd_trical.misc import constants as cst

########################################################################################
from .base import Base

########################################################################################


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
            grad_phi_x = np.empty(self.N * self.dim)

            i = 0
            for var in itr.product(
                ["x", "y", "z"][: self.dim], np.arange(self.N, dtype=int)
            ):
                grad_phi_x[i] = self.dphi(var)(x)
                i += 1
            return grad_phi_x

        return grad_phi

    def hessian(self):
        """
        Calculates the Hessian of the potential.

        Returns:
            (Callable): Function corresponding to the Hessian of the potential.
        """

        def hess_phi(x):
            hess_phi_x = np.empty((self.N * self.dim, self.N * self.dim))

            i = 0
            for var1 in itr.product(
                ["x", "y", "z"][: self.dim], np.arange(self.N, dtype=int)
            ):
                j = 0
                for var2 in itr.product(
                    ["x", "y", "z"][: self.dim], np.arange(self.N, dtype=int)
                ):
                    hess_phi_x[i, j] = self.d2phi(var1, var2)(x)
                    j += 1
                i += 1
            return hess_phi_x

        return hess_phi

    def nondimensionalize(self, l):
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

        super(CoulombPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        i, j = (
            np.fromiter(itr.chain(*itr.combinations(range(self.N), 2)), dtype=int)
            .reshape(-1, 2)
            .transpose()
        )
        nxij = np.linalg.norm(x[i] - x[j], axis=-1)
        return cst.k_e * self.q**2 * (1 / nxij).sum()

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])
        j = np.delete(np.arange(self.N, dtype=int), i)

        def dphi_dai(x):
            xia = x[i, a]
            xja = x[j, a]
            nxij = np.linalg.norm(x[i] - x[j], axis=-1)
            return cst.k_e * self.q**2 * ((xja - xia) / nxij**3).sum()

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])

        def d2phi_daidbj(x):
            if i == j:
                k = np.delete(np.arange(self.N, dtype=int), i)
                xia = x[i, a]
                xka = x[k, a]
                xib = x[i, b]
                xkb = x[k, b]
                nxik = np.linalg.norm(x[i] - x[k], axis=-1)
                if a == b:
                    return (
                        cst.k_e
                        * self.q**2
                        * (-1 / nxik**3 + 3 * (xka - xia) ** 2 / nxik**5).sum()
                    )
                else:
                    return (
                        cst.k_e
                        * self.q**2
                        * (3 * (xka - xia) * (xkb - xib) / nxik**5).sum()
                    )
            else:
                xia = x[i, a]
                xja = x[j, a]
                xib = x[i, b]
                xjb = x[j, b]
                nxij = np.linalg.norm(x[i] - x[j])
                if a == b:
                    return (
                        cst.k_e
                        * self.q**2
                        * (1 / nxij**3 - 3 * (xja - xia) ** 2 / nxij**5)
                    )
                else:
                    return (
                        cst.k_e * self.q**2 * (-3 * (xja - xia) * (xjb - xib) / nxij**5)
                    )

        return d2phi_daidbj

    def nondimensionalize(self, l):
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
        return {1: poly.polyval, 2: poly.polyval2d, 3: poly.polyval3d}[self.dim](
            *x.transpose(), self.alpha
        ).sum()

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])

        beta = poly.polyder(self.alpha, axis=a)

        def dphi_dai(x):
            return {1: poly.polyval, 2: poly.polyval2d, 3: poly.polyval3d}[self.dim](
                *x[i], beta
            )

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])

        beta = poly.polyder(self.alpha, axis=a)
        gamma = poly.polyder(beta, axis=b)

        if i == j:

            def d2phi_daidbj(x):
                return {
                    1: poly.polyval,
                    2: poly.polyval2d,
                    3: poly.polyval3d,
                }[self.dim](*x[i], gamma)
        else:

            def d2phi_daidbj(x):
                return 0.0

        return d2phi_daidbj

    def nondimensionalize(self, l):
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
            "m": cst.convert_m_a(171),
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
        I = 2 * power / (np.pi * beam_waist**2)
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
        delta_x = x - self.focal_point
        w0 = self.beam_waist
        w = w0 * np.sqrt(1 + (delta_x[:, 0] / self.x_R) ** 2)
        V = self.V
        r = np.sqrt(delta_x[:, 1] ** 2 + delta_x[:, 2] ** 2)
        e = np.exp(-2 * r**2 / w**2)
        return (V * e * w0**2 / w**2).sum()

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])

        def dphi_dai(x):
            V = self.V
            w0 = self.beam_waist
            xR = self.x_R
            delta_x = x[i] - self.focal_point
            w = w0 * np.sqrt(1 + (delta_x[0] / xR) ** 2)
            r = np.sqrt(delta_x[1] ** 2 + delta_x[2] ** 2)
            e = np.exp(-2 * r**2 / w**2)
            if a == 0:
                return (2 * V * e * w0**4 * delta_x[0] * (2 * r**2 - w**2)) / (
                    w**6 * xR**2
                )
            else:
                return -4 * V * e * w0**2 * delta_x[a] / w**4

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])

        def d2phi_daidbj(x):
            V = self.V
            w0 = self.beam_waist
            xR = self.x_R
            delta_x = x[i] - self.focal_point
            w = w0 * np.sqrt(1 + (delta_x[0] / xR) ** 2)
            r = np.sqrt(delta_x[1] ** 2 + delta_x[2] ** 2)
            e = np.exp(-2 * r**2 / w**2)
            if i != j:
                return 0
            else:
                if a == b == 0:
                    return (
                        -2
                        * V
                        * w0**4
                        * (
                            w**6 * xR**2
                            - 4 * w**4 * w0**2 * delta_x[0] ** 2
                            + 8 * w**2 * w0**2 * delta_x[0] ** 2 * r**2
                            - 2
                            * r**2
                            * (
                                w**4 * xR**2
                                - 4 * w**2 * w0**2 * delta_x[0] ** 2
                                + 4 * w0**2 * delta_x[0] ** 2 * r**2
                            )
                        )
                        * e
                        / (w**10 * xR**4)
                    )
                elif a == b:
                    return -4 * V * w0**2 * (w**2 - 4 * delta_x[a] ** 2) * e / w**6

                elif a == 0:
                    return (
                        16
                        * V
                        * w0**4
                        * delta_x[0]
                        * delta_x[b]
                        * (w**2 - r**2)
                        * e
                        / (w**8 * xR**2)
                    )
                elif b == 0:
                    return (
                        16
                        * V
                        * w0**4
                        * delta_x[0]
                        * delta_x[a]
                        * (w**2 - r**2)
                        * e
                        / (w**8 * xR**2)
                    )
                else:
                    return 16 * V * w0**2 * delta_x[1] * delta_x[2] * e / w**6

        return d2phi_daidbj

    def nondimensionalize(self, l):
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
        self.lambdified_expr = sympy.utilities.lambdify(self.symbol, expr)

        super(SymbolicPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        return self.lambdified_expr(*x.transpose()).sum()

    def evaluate(self, x):
        return self.lambdified_expr(*x.transpose())

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])

        def dphi_dai(x):
            return sympy.utilities.lambdify(
                self.symbol, sympy.diff(self.expr, self.symbol[a])
            )(*x[i])

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])

        if i == j:

            def d2phi_daidbj(x):
                return sympy.utilities.lambdify(
                    self.symbol, sympy.diff(self.expr, self.symbol[a], self.symbol[b])
                )(*x[i])
        else:

            def d2phi_daidbj(x):
                return 0

        return d2phi_daidbj

    def nondimensionalize(self, l):
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
        self.lambdified_expr = sympy.utilities.lambdify(self.symbol, expr)

        super(AdvancedSymbolicPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        x = np.array(x)
        return self.lambdified_expr(*x.flatten())

    def first_derivative(self, var):
        a = var[0]
        i = int(var[1:] if isinstance(var, str) else var[1:][0])

        def dphi_dai(x):
            x = np.array(x)
            return sympy.utilities.lambdify(
                self.symbol, sympy.diff(self.expr, a + str(i))
            )(*x.flatten())

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = var1[0]
        b = var2[0]
        i = int(var1[1:] if isinstance(var1, str) else var1[1:][0])
        j = int(var2[1:] if isinstance(var2, str) else var2[1:][0])

        def d2phi_daidbj(x):
            x = np.array(x)
            return sympy.utilities.lambdify(
                self.symbol, sympy.diff(self.expr, a + str(i), b + str(j))
            )(*x.flatten())

        return d2phi_daidbj

    def nondimensionalize(self, l):
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
            "m": cst.convert_m_a(171),
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
        expr (Callable): function of the potential that is defined using the numpy submodule of autograd package.

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

        return lambda x: ag.jacobian(flatten_expr, 0)(x.transpose().reshape(-1))

    def hessian(self):
        def flatten_expr(x):
            return self.expr(x.reshape(self.dim, -1).transpose())

        return lambda x: ag.hessian(flatten_expr, 0)(x.transpose().reshape(-1))

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

    def nondimensionalize(self, l):
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
        intensity_expr (Callable): function of the expression for intensity of the optical potential that is defined using the numpy submodule of autograd package.
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
            "m": cst.convert_m_a(171),
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
