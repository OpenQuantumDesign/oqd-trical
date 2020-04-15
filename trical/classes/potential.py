from ..misc import constants as cst
from ..misc.linalg import norm
from ..misc.polynomial import multivariate_polyfit
from ..misc.setalg import intersection
import itertools as itr
import numpy as np
from numpy.polynomial import polynomial as poly
import sympy


class Potential(object):
    """
    Object representing a general potential.

    :param d2phi: Function that takes two strings representing the derivative variables and outputs the function corresponding to the derivative of the potential with respect to the derivative variables.
    :type d2phi: :obj:`types.FunctionType`
    :param dphi: Function that takes a string representing the derivative variable and outputs the function corresponding to the derivative of the potential with respect to the derivative variable.
    :type dphi: :obj:`types.FunctionType`
    :param phi: Function representing the potential.
    :type phi: :obj:`types.FunctionType`
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
        for i in intersection(self.params.keys(), other.params.keys()):
            assert self.params[i] == other.params[i]

        params = {}
        params.update(self.params)
        params.update(other.params)
        phi = lambda x: self.phi(x) + other.phi(x)
        dphi = lambda var: (lambda x: self.dphi(var)(x) + other.dphi(var)(x))
        d2phi = lambda var1, var2: (
            lambda x: self.d2phi(var1, var2)(x) + other.d2phi(var1, var2)(x)
        )
        return Potential(phi, dphi, d2phi, **params)

    def __sub__(self, other):
        for i in intersection(self.params.keys(), other.params.keys()):
            assert self.params[i] == other.params[i]

        params = {}
        params.update(self.params)
        params.update(other.params)
        phi = lambda x: self.phi(x) - other.phi(x)
        dphi = lambda var: (lambda x: self.dphi(var)(x) - other.dphi(var)(x))
        d2phi = lambda var1, var2: (
            lambda x: self.d2phi(var1, var2)(x) - other.d2phi(var1, var2)(x)
        )
        return Potential(phi, dphi, d2phi, **params)

    def __mul__(self, multiplier):
        phi = lambda x: self.phi(x) * multiplier
        dphi = lambda var: (lambda x: self.dphi(var)(x) * multiplier)
        d2phi = lambda var1, var2: (lambda x: self.d2phi(var1, var2)(x) * multiplier)
        return Potential(phi, dphi, d2phi, **self.params)

    def __rmul__(self, multiplier):
        return self * multiplier

    def __truediv__(self, divisor):
        phi = lambda x: self.phi(x) / divisor
        dphi = lambda var: (lambda x: self.dphi(var)(x) / divisor)
        d2phi = lambda var1, var2: (lambda x: self.d2phi(var1, var2)(x) / divisor)
        return Potential(phi, dphi, d2phi, **self.params)

    def __call__(self, x):
        return self.phi(x)

    def first_derivative(self, var):
        """
        Calculates the first derivative of the potential with respect to a variable.

        :param var: Derivative variable.
        :type var: :obj:`str`
        :returns: Function corresponding to the first derivative of the potential with respect to the derivative variable.
        :rtype: :obj:`types.FunctionType`
        """
        return self.dphi(var)

    def second_derivative(self, var1, var2):
        """
        Calculates the second derivative of the potential with respect to two variables.

        :param var1: first derivative variable.
        :type var1: :obj:`str`
        :param var2: second derivative variable.
        :type var2: :obj:`str`
        :returns: Function corresponding to the second derivative of the potential with respect to the derivative variables.
        :rtype: :obj:`types.FunctionType`
        """
        return self.d2phi(var1, var2)

    def gradient(self):
        """
        Calculates the gradient of the potential

        :returns: Function corresponding to the gradient of the potential
        :rtype: :obj:`types.FunctionType`
        """

        def grad_phi(x):
            grad_phi_x = np.empty(self.N * self.dim)

            i = 0
            for var in itr.product(
                ["x", "y", "z"][: self.dim], np.arange(1, self.N + 1, dtype=int)
            ):
                grad_phi_x[i] = self.dphi(var)(x)
                i += 1
            return grad_phi_x

        return grad_phi

    def hessian(self):
        """
        Calculates the Hessian of the potential

        :returns: Function corresponding to the Hessian of the potential
        :rtype: :obj:`types.FunctionType`
        """

        def hess_phi(x):
            hess_phi_x = np.empty((self.N * self.dim, self.N * self.dim))

            i = 0
            for var1 in itr.product(
                ["x", "y", "z"][: self.dim], np.arange(1, self.N + 1, dtype=int)
            ):
                j = 0
                for var2 in itr.product(
                    ["x", "y", "z"][: self.dim], np.arange(1, self.N + 1, dtype=int)
                ):
                    hess_phi_x[i, j] = self.d2phi(var1, var2)(x)
                    j += 1
                i += 1
            return hess_phi_x

        return hess_phi

    def update_params(self, **kwargs):
        """
        Updates parameters, i.e. params attribute, of a Potential object.

        :Keyword Arguments:
            * **dim** (:obj:`float`): Dimension of the system.
            * **m** (:obj:`float`): Mass of an ion.
            * **N** (:obj:`float`): Number of Ions.
            * **q** (:obj:`dict`): Charge of an ion.
        """
        self.params.update(kwargs)
        self.__dict__.update(self.params)
        pass

    pass


class CoulombPotential(Potential):
    """
    Object representing a coulomb potential.

    :param N: Number of ions.
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
        nxij = norm(x[i] - x[j])
        return cst.k * self.q ** 2 * (1 / nxij).sum()

    def first_derivative(self, var):
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if type(var) == str else var[1:][0]) - 1
        j = np.delete(np.arange(self.N, dtype=int), i)

        def dphi_dai(x):
            xia = x[i, a]
            xja = x[j, a]
            nxij = norm(x[i] - x[j])
            return cst.k * self.q ** 2 * ((xja - xia) / nxij ** 3).sum()

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if type(var1) == str else var1[1:][0]) - 1
        j = int(var2[1:] if type(var2) == str else var2[1:][0]) - 1

        def d2phi_daidbj(x):
            if i == j:
                k = np.delete(np.arange(self.N, dtype=int), i)
                xia = x[i, a]
                xka = x[k, a]
                xib = x[i, b]
                xkb = x[k, b]
                nxik = norm(x[i] - x[k])
                if a == b:
                    return (
                        cst.k
                        * self.q ** 2
                        * ((-1 / nxik ** 3 + 3 * (xka - xia) ** 2 / nxik ** 5)).sum()
                    )
                else:
                    return (
                        cst.k
                        * self.q ** 2
                        * (3 * (xka - xia) * (xkb - xib) / nxik ** 5).sum()
                    )
            else:
                xia = x[i, a]
                xja = x[j, a]
                xib = x[i, b]
                xjb = x[j, b]
                nxij = norm(x[i] - x[j])
                if a == b:
                    return (
                        cst.k
                        * self.q ** 2
                        * (1 / nxij ** 3 - 3 * (xja - xia) ** 2 / nxij ** 5)
                    )
                else:
                    return (
                        cst.k
                        * self.q ** 2
                        * (-3 * (xja - xia) * (xjb - xib) / nxij ** 5)
                    )

        return d2phi_daidbj

    def nondimensionalize(self, l):
        """
        Nondimensionalizes a CoulombPotential with a length scale.

        :param l: Length scale.
        :type l: :obj:`float`
        :returns: Potential representing the nondimensionalized coulomb potential.
        :rtype: :obj:`trical.classes.potential.Potential`
        """
        return self / (cst.k * cst.e ** 2)

    pass


class PolynomialPotential(Potential):
    """
    Object representing a polynomial potential.

    :param alpha: Coefficients of the polynomial potential.
    """

    def __init__(self, alpha):
        self.alpha = np.array(alpha)
        self.deg = np.array(alpha.shape) - 1

        params = {"dim": len(alpha.shape)}

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
        i = int(var[1:] if type(var) == str else var[1:][0]) - 1

        beta = poly.polyder(self.alpha, axis=a)

        dphi_dai = lambda x: {1: poly.polyval, 2: poly.polyval2d, 3: poly.polyval3d}[
            self.dim
        ](*x[i], beta)

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if type(var1) == str else var1[1:][0]) - 1
        j = int(var2[1:] if type(var2) == str else var2[1:][0]) - 1

        beta = poly.polyder(self.alpha, axis=a)
        gamma = poly.polyder(beta, axis=b)

        if i == j:
            d2phi_daidbj = lambda x: {
                1: poly.polyval,
                2: poly.polyval2d,
                3: poly.polyval3d,
            }[self.dim](*x[i], gamma)
        else:
            d2phi_daidbj = lambda x: 0.0

        return d2phi_daidbj

    def nondimensionalize(self, l):
        """
        Nondimensionalizes a PolynomialPotential with a length scale.

        :param l: Length scale.
        :type l: :obj:`float`
        :returns: Nondimensionalized PolynomialPotential.
        :rtype: :obj:`trical.classes.potential.PolynomialPotential`
        """
        alpha = (
            l ** np.indices(self.alpha.shape).sum(0)
            * self.alpha
            * (l / (cst.k * cst.e ** 2))
        )
        return PolynomialPotential(alpha)

    pass


class SymbolicPotential(Potential):
    """
    Object representing a symbolically defined potential, same for all ions.

    :param expr: Symbolic expression of the potential.
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
        i = int(var[1:] if type(var) == str else var[1:][0]) - 1

        dphi_dai = lambda x: sympy.utilities.lambdify(
            self.symbol, sympy.diff(self.expr, self.symbol[a])
        )(*x[i])

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if type(var1) == str else var1[1:][0]) - 1
        j = int(var2[1:] if type(var2) == str else var2[1:][0]) - 1

        if i == j:
            d2phi_daidbj = lambda x: sympy.utilities.lambdify(
                self.symbol, sympy.diff(self.expr, self.symbol[a], self.symbol[b])
            )(*x[i])
        else:
            d2phi_daidbj = lambda x: 0

        return d2phi_daidbj

    def nondimensionalize(self, l):
        """
        Nondimensionalizes a SymbolicPotential with a length scale.

        :param l: Length scale.
        :type l: :obj:`float`
        :returns: Nondimensionalized SymbolicPotential.
        :rtype: :obj:`trical.classes.potential.SymbolicPotential`
        """
        expr = self.expr.subs({k: k * l for k in self.symbol}) * (
            l / (cst.k * cst.e ** 2)
        )
        return SymbolicPotential(expr, **self.params)

    pass


class AdvancedSymbolicPotential(Potential):
    """
    Object representing a symbolically defined potential that need not be the same for all ions.

    :param expr: Symbolic expression of the potential.
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
        i = int(var[1:] if type(var) == str else var[1:][0]) - 1

        def dphi_dai(x):
            x = np.array(x)
            return sympy.utilities.lambdify(
                self.symbol, sympy.diff(self.expr, a + str(i))
            )(*x.flatten())

        return dphi_dai

    def second_derivative(self, var1, var2):
        a = var1[0]
        b = var2[0]
        i = int(var1[1:] if type(var1) == str else var1[1:][0]) - 1
        j = int(var2[1:] if type(var2) == str else var2[1:][0]) - 1

        def d2phi_daidbj(x):
            x = np.array(x)
            return sympy.utilities.lambdify(
                self.symbol, sympy.diff(self.expr, a + str(i), b + str(j))
            )(*x.flatten())

        return d2phi_daidbj

    def nondimensionalize(self, l):
        """
        Nondimensionalizes a AdvancedSymbolicPotential with a length scale.

        :param l: Length scale.
        :type l: :obj:`float`
        :returns: Nondimensionalized AdvancedSymbolicPotential.
        :rtype: :obj:`trical.classes.potential.AdvancedSymbolicPotential`
        """
        expr = self.expr.subs({k: k * l for k in self.symbol}) * (
            l / (cst.k * cst.e ** 2)
        )
        params = self.params
        if "N" in params.keys():
            params.pop("N")
        return AdvancedSymbolicPotential(self.N, expr, **params)

    pass


class OpticalPotential(SymbolicPotential):
    """
    Object representing a potential caused by a Gaussian beam.

    :param focal_point: Center of the Gaussian beam.
    :type focal_point: :obj:`numpy.ndarray`
    :param power: Power of Gaussian beam.
    :type power: :obj:`float`
    :param wavelength: Wavelength of Gaussian beam.
    :type wavelength: :obj:`float`
    :param beam_waist: Waist of Gaussian beam.
    :type beam_waist: :obj:`float`

    :Keyword Arguments:
        * **m** (:obj:`float`): Mass of ion.
        * **rfpri** (:obj:`float`): Rabi frequency per root intensity.
        * **transition_wavelength** (:obj:`float`): Wavelength of the transition that creates the optical trap.
        * **refractive_index** (:obj:`float`): Refractive index of medium Gaussian beam is propagating through.
    """

    def __init__(self, focal_point, power, wavelength, beam_waist, **kwargs):
        c = 2.99792458e8

        self.params = {"dim": 3}

        self.focal_point = focal_point
        self.power = power
        self.wavelength = wavelength
        self.beam_waist = beam_waist

        opt_params = {
            "m": cst.m_a["Yb171"],
            "rfpri": 3.86e6,
            "transition_wavelength": 369.52e-9,
            "refractive_index": 1,
        }
        opt_params.update(kwargs)
        self.__dict__.update(opt_params)

        x_sym, y_sym, z_sym = sympy.symbols("x y z")

        delta_x, delta_y, delta_z = (
            x_sym - focal_point[0],
            y_sym - focal_point[1],
            z_sym - focal_point[2],
        )
        Delta = (
            c * 2 * np.pi * (1 / wavelength - 1 / opt_params["transition_wavelength"])
        )
        x_R = np.pi * beam_waist ** 2 * opt_params["refractive_index"] / wavelength
        w = beam_waist * sympy.sqrt(1 + (delta_x / x_R) ** 2)
        I = 2 * power / (np.pi * beam_waist ** 2)

        self.Delta = Delta
        self.x_R = x_R
        self.I = I

        expr = (
            cst.hbar
            * opt_params["rfpri"] ** 2
            * I
            * beam_waist ** 2
            * sympy.exp(-2 * (delta_y ** 2 + delta_z ** 2) / w ** 2)
            / (w ** 2 * 4 * Delta)
        )

        super(OpticalPotential, self).__init__(expr, dim=3)

        omega_x = np.sqrt(
            np.abs(
                cst.hbar
                * opt_params["rfpri"] ** 2
                * power
                * wavelength ** 2
                / (
                    2
                    * opt_params["refractive_index"] ** 2
                    * np.pi ** 3
                    * Delta
                    * beam_waist ** 6
                )
            )
            * 2
            / self.m
        )
        omega_y = omega_z = np.sqrt(
            np.abs(
                cst.hbar
                * opt_params["rfpri"] ** 2
                * power
                / (np.pi * Delta * beam_waist ** 4)
            )
            * 2
            / self.m
        )
        self.omega = np.array([omega_x, omega_y, omega_z])
        pass

    def fit_trap_strength(self, ls_to_bw=10):
        """
        """
        x = np.moveaxis(
            np.meshgrid(
                *(
                    [
                        np.linspace(
                            -self.beam_waist / ls_to_bw, self.beam_waist / ls_to_bw, 101
                        )
                    ]
                    * 3
                )
            ),
            (0, 1, 2, 3),
            (3, 0, 1, 2),
        ).reshape(-1, 3)

        V = self.evaluate(x + self.focal_point)

        alpha = multivariate_polyfit(x, V, deg=(2, 2, 2), l=self.beam_waist / 100)

        self.omega_fit = np.sqrt(alpha[tuple(np.eye(3, dtype=int) * 2)] * 2 / self.m)
        pass

    pass
