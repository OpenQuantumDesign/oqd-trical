"""
Defines the Potential class and its subclasses representing potentials associated with a trapped ions system
"""
from .. import constants as cst
from ..misc.linalg import norm
from ..misc.setalg import intersection
import itertools as itr
import numpy as np
from numpy.polynomial import polynomial as poly
import sympy


class Potential(object):

    """
    Object representing a general potential

    Attributes:
        d2phi (func(str,str) -> (func(1-D or 2-D array of float) -> float)): Function that takes two strings representing the derivative variables and outputs the function corresponding to the derivative of the potential with respect to the derivative variables
        dphi (func(str) -> (func(1-D or 2-D array of float) -> float)): Function that takes a string representing the derivative variable and outputs the function corresponding to the derivative of the potential with respect to the derivative variable
        phi (func(1-D or 2-D array of float)): Function representing the potential
        params (dict): Other relevant parameters of the Potential object
    """

    def __init__(self, phi, dphi, d2phi, **kwargs):
        """
        Initialization function for a Potential object

        Args:
            phi (func(1-D or 2-D array of float)): Function representing the potential
            dphi (func(str,str) -> (func(1-D or 2-D array of float) -> float)): Function that takes a string representing the derivative variable and outputs the function corresponding to the derivative of the potential with respect to the derivative variable
            d2phi (func(str) -> (func(1-D or 2-D array of float) -> float)): Function that takes two strings representing the derivative variables and outputs the function corresponding to the derivative of the potential with respect to the derivative variables

        Kwargs:
            dim (int, optional): Dimension of the system
            m (float, optional): Mass of an ion
            N (int, optional): Number of Ions
            q (float, optional): Charge of an ion

        """
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
        """
        Adds two Potential objects

        Args:
            other (Potential): Potential object of interest

        Returns:
            Potential: Addition of the potentials
        """
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
        """
        Subtracts two Potential objects

        Args:
            other (Potential): Potential object of interest

        Returns:
            Potential: Subtraction of the potentials
        """
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
        """
        Multiplies Potential objects by some multiplier

        Args:
            multiplier (float): Scalar to multiply the potential by

        Returns:
            Potential: The potential multiplied by the multiplier
        """
        phi = lambda x: self.phi(x) * multiplier
        dphi = lambda var: (lambda x: self.dphi(var)(x) * multiplier)
        d2phi = lambda var1, var2: (lambda x: self.d2phi(var1, var2)(x) * multiplier)
        return Potential(phi, dphi, d2phi, **self.params)

    def __rmul__(self, multiplier):
        """
        Multiplies Potential objects by some multiplier

        Args:
            multiplier (float): Scalar to multiply the potential by

        Returns:
            Potential: The potential multiplied by the multiplier
        """
        return self * multiplier

    def __truediv__(self, divisor):
        """
        Divides Potential objects by some divisor

        Args:
            divisor (float): Scalar to divide the potential by

        Returns:
            Potential: The potential divided by the divisor
        """
        phi = lambda x: self.phi(x) / divisor
        dphi = lambda var: (lambda x: self.dphi(var)(x) / divisor)
        d2phi = lambda var1, var2: (lambda x: self.d2phi(var1, var2)(x) / divisor)
        return Potential(phi, dphi, d2phi, **self.params)

    def __call__(self, x):
        """
        Evaluates the potential given the position of the ions

        Args:
            x (1-D or 2-D array of floats): Position of the ions

        Returns:
            float: value of the potential given the position of the ions
        """
        return self.phi(x)

    def first_derivative(self, var):
        """
        Calculates the first derivative of the potential with respect to a variable

        Args:
            var (str): Derivative variable

        Returns:
            func(1-D or 2-D array of float) -> float: Function corresponding to the first derivative of the potential with respect to the derivative variable
        """
        return self.dphi(var)

    def second_derivative(self, var1, var2):
        """
        Calculates the second derivative of the potential with respect to two variables

        Args:
            var1 (str): first derivative variable
            var2 (str): second derivative variable

        Returns:
            func(1-D or 2-D array of float) -> float: Function corresponding to the second derivative of the potential with respect to the derivative variables
        """
        return self.d2phi(var1, var2)

    def gradient(self):
        """
        Calculates the gradient of the potential

        Returns:
            func(1-D or 2-D array of float) -> 1-D array of float: Function corresponding to the gradient of the potential
        """

        def grad_phi(x):
            """
            Function corresponding to the gradient of the potential

            Args:
                x (1-D or 2-D array of float): Position of the ions

            Returns:
                1-D array of float: Value of the gradient of the potential given the position of the ions
            """
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

        Returns:
            func(1-D or 2-D array of float) -> 2-D array of float: Function corresponding to the Hessian of the potential
        """

        def hess_phi(x):
            """
            Function corresponding to the Hessian of the potential

            Args:s
                x (1-D or 2-D array of float): Position of the ions

            Returns:
                1-D array of float: Value of the Hessian of the potential given the position of the ions
            """
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
        Updates parameters of Potential object

        Kwargs:
            dim (int, optional): Dimension of the system
            m (float, optional): Mass of an ion
            N (int, optional): Number of Ions
            q (float, optional): Charge of an ion

        """
        self.params.update(kwargs)
        self.__dict__.update(self.params)
        pass

    pass


class CoulombPotential(Potential):

    """
    Object representing a Coulomb potential
    """

    def __init__(self, N, **kwargs):
        """
        Initialization function for a CoulombPotential object

        Args:
            N (int): Number of ions

        Kwargs:
            N (int): Number of Ions
            dim (int, optional): Dimension of the system
            q (float, optional): Charge of an ion
        """
        params = {"dim": 3, "N": N, "q": cst.e}
        params.update(kwargs)

        super(CoulombPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        """
        Evaluates the Coulomb potential given the position of the ions

        Args:
            x (1-D or 2-D array of floats): Position of the ions

        Returns:
            float: value of the Coulomb potential given the position of the ions
        """
        i, j = (
            np.fromiter(itr.chain(*itr.combinations(range(self.N), 2)), dtype=int)
            .reshape(-1, 2)
            .transpose()
        )
        nxij = norm(x[i] - x[j])
        return cst.k * self.q ** 2 * (1 / nxij).sum()

    def first_derivative(self, var):
        """
        Calculates the first derivative of the Coulomb potential with respect to a variable

        Args:
            var (str): Derivative variable

        Returns:
            func(1-D or 2-D array of float) -> float: Function corresponding to the first derivative of the Coulomb potential with respect to the derivative variable
        """
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if type(var) == str else var[1:][0]) - 1
        j = np.delete(np.arange(self.N, dtype=int), i)

        def dphi_dai(x):
            """
            Function corresponding to a first derivative of the Coulomb potential

            Args:
                x (1-D or 2-D array of float): Position of the ions

            Returns:
                float: Value of a first derivative of the Coulomb potential given the position of the ions
            """
            xia = x[i, a]
            xja = x[j, a]
            nxij = norm(x[i] - x[j])
            return cst.k * self.q ** 2 * ((xja - xia) / nxij ** 3).sum()

        return dphi_dai

    def second_derivative(self, var1, var2):
        """
        Calculates the second derivative of the Coulomb potential with respect to two variables

        Args:
            var1 (str): first derivative variable
            var2 (str): second derivative variable

        Returns:
            func(1-D or 2-D array of float) -> float: Function corresponding to the second derivative of the Coulomb potential with respect to the derivative variables
        """
        a = {"x": 0, "y": 1, "z": 2}[var1[0]]
        b = {"x": 0, "y": 1, "z": 2}[var2[0]]
        i = int(var1[1:] if type(var1) == str else var1[1:][0]) - 1
        j = int(var2[1:] if type(var2) == str else var2[1:][0]) - 1

        def d2phi_daidbj(x):
            """
            Function corresponding to a second derivative of the Coulomb potential

            Args:s
                x (1-D or 2-D array of float): Position of the ions

            Returns:
                float: Value of a second derivative of the Coulomb potential given the position of the ions
            """
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
        Nondimensionalizes the Coulomb potential using a length scale

        Args:
            l (float): A length scale

        Returns:
            Potential: Nondimensionalized Coulomb potential
        """
        return self / (cst.k * cst.e ** 2)

    pass


class PolynomialPotential(Potential):

    """
    Object representing a Polynomial potential

    Attributes:
        alpha (1-D, 2-D or 3-D array of float): Polynomial coefficients
        deg (int or 1-D array of int): Degree of Polynomial
        dim (int, optional): Dimension of the system
    """

    def __init__(self, alpha):
        """
        Initialization function for a PolynomialPotential object

        Args:
            alpha (1-D, 2-D or 3-D array of float): Polynomial coefficients
        """
        self.alpha = np.array(alpha)
        self.deg = np.array(alpha.shape) - 1

        params = {"dim": len(alpha.shape)}

        super(PolynomialPotential, self).__init__(
            self.__call__, self.first_derivative, self.second_derivative, **params
        )
        pass

    def __call__(self, x):
        """
        Evaluates the polynomial potential given the position of the ions

        Args:
            x (1-D or 2-D array of floats): Position of the ions

        Returns:
            float: value of the polynomial potential given the position of the ions
        """
        return {1: poly.polyval, 2: poly.polyval2d, 3: poly.polyval3d}[self.dim](
            *x.transpose(), self.alpha
        ).sum()

    def first_derivative(self, var):
        """
        Calculates the first derivative of the polynomial potential with respect to a variable

        Args:
            var (str): Derivative variable

        Returns:
            func(1-D or 2-D array of float) -> float: Function corresponding to the first derivative of the polynomial potential with respect to the derivative variable
        """
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if type(var) == str else var[1:][0]) - 1

        beta = poly.polyder(self.alpha, axis=a)

        dphi_dai = lambda x: {1: poly.polyval, 2: poly.polyval2d, 3: poly.polyval3d}[
            self.dim
        ](*x[i], beta)

        return dphi_dai

    def second_derivative(self, var1, var2):
        """
        Calculates the second derivative of the polynomial potential with respect to two
        variables

        Args:
            var1 (str): first derivative variable
            var2 (str): second derivative variable

        Returns:
            func(1-D or 2-D array of float) -> float: Function corresponding to the second derivative of the polynomial potential with respect to the derivative variables
        """
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
        Nondimensionalizes the polynomial potential using a length scale

        Args:
            l (float): A length scale

        Returns:
            PolynomialPotential: Nondimensionalized polynomial potential
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
    Object representing a symbolically defined potential

    Attributes:
        expr (sympy): Symbolic expression of the potential
        dim (int, optional): Dimension of the system
    """

    def __init__(self, expr, **kwargs):
        """
        Initialization function for a SymbolicPotential object

        Args:
            expr (sympy): Symbolic expression of the potential
        Kwargs:
            dim (int, optional): Dimension of the system
        """
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
        """
        Evaluates the symbolically defined potential given the position of the ions

        Args:
            x (1-D or 2-D array of floats): Position of the ions

        Returns:
            float: value of the symbolically defined potential given the position of the ions
        """
        return self.lambdified_expr(*x.transpose()).sum()

    def first_derivative(self, var):
        """
        Calculates the first derivative of the symbolically defined potential with respect to a variable

        Args:
            var (str): Derivative variable

        Returns:
            func(1-D or 2-D array of float) -> float: Function corresponding to the first derivative of the symbolically defined potential with respect to the derivative variable
        """
        a = {"x": 0, "y": 1, "z": 2}[var[0]]
        i = int(var[1:] if type(var) == str else var[1:][0]) - 1

        dphi_dai = lambda x: sympy.utilities.lambdify(
            self.symbol, sympy.diff(self.expr, self.symbol[a])
        )(*x[i])

        return dphi_dai

    def second_derivative(self, var1, var2):
        """
        Calculates the second derivative of the symbolically defined potential with respect to two variables

        Args:
            var1 (str): first derivative variable
            var2 (str): second derivative variable

        Returns:
            func(1-D or 2-D array of float) -> float: Function corresponding to the second derivative of the symbolically defined potential with respect to the derivative variables
        """
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
        Nondimensionalizes the symbolically defined potential using a length scale

        Args:
            l (float): A length scale

        Returns:
            PolynomialPotential: Nondimensionalized symbolically defined potential
        """
        expr = self.expr.subs({k: k * l for k in self.symbol}) * (
            l / (cst.k * cst.e ** 2)
        )
        return SymbolicPotential(expr, **self.params)

    pass
