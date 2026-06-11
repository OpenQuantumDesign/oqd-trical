# Copyright 2024-2026 Open Quantum Design

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
import jax
import jax.numpy as jnp
import numpy as np

import oqd_trical
import oqd_trical.misc.constants as cst
from oqd_trical.misc.polynomial import polyval

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)


def assert_eigenvectors_equal_up_to_phase(U, V, rtol=1e-6, atol=5e-8):
    U = np.asarray(U)
    V = np.asarray(V)

    inner_products = np.sum(np.conj(V) * U, axis=0)

    mags = np.abs(inner_products)
    phase_factors = np.where(mags > 1e-10, inner_products / mags, 1.0)

    V_aligned = V * phase_factors

    # 4. Compare
    np.testing.assert_allclose(U, V_aligned, rtol=rtol, atol=atol)


def advanced_symbolic_potential_ti():
    import sympy as sym

    N = 3  # Number of ions
    dim = 3  # Dimension of system

    mass = 171 * cst.m_u  # Mass of an ion

    # Trapping strength (in rad/s)
    omega_x = 2 * np.pi * 0.4e6  # Direction x
    omega_y = 2 * np.pi * 0.36e6  # Direction y
    omega_z = 2 * np.pi * 0.08e6  # Direction z

    # Symbols for the coordinates of the different ions
    x = np.array(sym.symbols(" ".join(["x{}".format(i) for i in range(N)])))
    y = np.array(sym.symbols(" ".join(["y{}".format(i) for i in range(N)])))
    z = np.array(sym.symbols(" ".join(["z{}".format(i) for i in range(N)])))

    # Symbolic expression for the potential
    expr = (
        (mass * omega_x**2 * x**2 / 2).sum()
        + (mass * omega_y**2 * y**2 / 2).sum()
        + (mass * omega_z**2 * z**2 / 2).sum()
    )

    # Define an instance of the AdvancedSymbolicPotential class for a harmonic potential
    asp = oqd_trical.mechanical.AdvancedSymbolicPotential(N, expr, dim=dim)
    ti_asp = oqd_trical.mechanical.TrappedIons(
        N, asp, m=mass
    )  # AdvancedSymbolicPotential
    return ti_asp


def optical_potential_ti():
    from jax import numpy as jnp

    N = 3  # Number of ions
    dim = 3  # Dimension of system
    mass = 171 * cst.m_u  # Mass of an ion

    # Parameters of the gaussian beam (in SI units)
    focal_point = np.zeros(3)  # Center of system
    beam_waist = 1e-6
    refractive_index = 1.0
    wavelength = 375e-9
    power = 1e0
    Omega_bar = 2.23e6
    transition_wavelength = 369.52e-9

    # Trapping strength (in rad/s)
    omega_x = 2 * np.pi * 0.4e6  # Direction x
    omega_y = 2 * np.pi * 0.36e6  # Direction y
    omega_z = 2 * np.pi * 0.08e6  # Direction z

    # function for the potential defined using autograd.numpy
    def expr(x):
        return np.sum(
            mass * (omega_x) ** 2 / 2 * x[:, 0] ** 2
            + mass * (omega_y) ** 2 / 2 * x[:, 1] ** 2
            + mass * (omega_z) ** 2 / 2 * x[:, 2] ** 2
        )

    # Define an instance of the AutoDiffPotential class for a harmonic potential
    adp = oqd_trical.mechanical.AutoDiffPotential(expr, N=N, dim=dim)

    # Parameters for a Gaussian beam
    focal_point = np.zeros(3)
    beam_waist = 1e-6
    refractive_index = 1.0
    wavelength = 375e-9
    power = 1e0
    Omega_bar = 2.23e6
    transition_wavelength = 369.52e-9

    # Function for the intensity for the Gaussian beam with the above parameters defined using autograd.numpy
    def intensity_expr(x):
        delta_x = x - focal_point
        x_R = np.pi * beam_waist**2 * refractive_index / wavelength
        w = beam_waist * jnp.sqrt(1 + (delta_x[:, 0] / x_R) ** 2)
        return (
            2
            * power
            / (np.pi * beam_waist**2)
            * (beam_waist / w) ** 2
            * jnp.exp(-2 * (delta_x[:, 1] ** 2 + delta_x[:, 2] ** 2) / w**2)
        ).sum()

    # Define an instance of the OpticalPotential class for the Gaussian beam defined above
    op = oqd_trical.mechanical.OpticalPotential(
        intensity_expr,
        wavelength,
        Omega_bar=Omega_bar,
        transition_wavelength=transition_wavelength,
    )

    op.update_params(N=N)  # Without N cannot create gradient or Hessian

    ti_op = oqd_trical.mechanical.TrappedIons(N, adp, op, m=mass)  # OpticalPotential
    return ti_op


def ms_potential():
    N = 3  # Number of ions
    dim = 3  # Dimension of system

    myb = 171 * cst.m_u  # mass of ytterbium-171
    mba = 138 * cst.m_u  # mass of barium-138
    ms = np.array(
        [myb, mba, myb]
    )  # mass array (length must be equal to number of ions)

    # Trapping strength (in rad/s)
    omega_x = 2 * np.pi * 0.4e6  # Direction x
    omega_y = 2 * np.pi * 0.36e6  # Direction y
    omega_z = 2 * np.pi * 0.08e6  # Direction z

    # Expression for ion dependent multispecies harmonic potential
    def expr(x):
        return (
            (myb**2 * omega_x**2 * x[:, 0] ** 2 / (2 * ms)).sum()
            + (myb**2 * omega_y**2 * x[:, 1] ** 2 / (2 * ms)).sum()
            + (myb * omega_z**2 * x[:, 2] ** 2 / 2).sum()
        )

    # AutoDiffPotential for above defined ion dependent multispecies harmonic potential
    adp = oqd_trical.mechanical.AutoDiffPotential(expr, N=N, dim=dim)
    return adp


def ms_potential_ti():
    N = 3  # Number of ions
    myb = 171 * cst.m_u  # mass of ytterbium-171
    mba = 138 * cst.m_u  # mass of barium-138
    ms = np.array(
        [myb, mba, myb]
    )  # mass array (length must be equal to number of ions)
    # Multispecies harmonic trapped ion system
    adp = ms_potential()
    ti_ms = oqd_trical.mechanical.TrappedIons(N, adp, m=ms)
    return ti_ms


def gaussian_optical_potential_ti():
    from jax import numpy as jnp

    N = 3  # Number of ions
    dim = 3  # Dimension of system
    mass = 171 * cst.m_u  # Mass of an ion

    # Parameters of the gaussian beam (in SI units)
    focal_point = np.zeros(3)  # Center of system
    beam_waist = 1e-6
    refractive_index = 1.0
    wavelength = 375e-9
    power = 1e0
    Omega_bar = 2.23e6
    transition_wavelength = 369.52e-9

    # Trapping strength (in rad/s)
    omega_x = 2 * np.pi * 0.4e6  # Direction x
    omega_y = 2 * np.pi * 0.36e6  # Direction y
    omega_z = 2 * np.pi * 0.08e6  # Direction z

    # function for the potential defined using autograd.numpy
    def expr(x):
        return jnp.sum(
            mass * (omega_x) ** 2 / 2 * x[:, 0] ** 2
            + mass * (omega_y) ** 2 / 2 * x[:, 1] ** 2
            + mass * (omega_z) ** 2 / 2 * x[:, 2] ** 2
        )

    # Define an instance of the AutoDiffPotential class for a harmonic potential
    adp = oqd_trical.mechanical.AutoDiffPotential(expr, N=N, dim=dim)

    # Define an instance of the GaussianOpticalPotential class
    gop = oqd_trical.mechanical.GaussianOpticalPotential(
        focal_point,
        power,
        wavelength,
        beam_waist,
        mass=mass,
        Omega_bar=Omega_bar,
        transition_wavelength=transition_wavelength,
        refractive_index=refractive_index,
    )

    # Assign a number of ions to the GaussianOpticalPotential class
    gop.update_params(N=N)  # Without N cannot create gradient or Hessian

    # Harmonic trapped ion system with optical tweezers on the center ion defined using different methods
    ti_gop = oqd_trical.mechanical.TrappedIons(
        N, adp, gop, m=mass
    )  # GaussianOpticalPotential
    return ti_gop


def symbolic_optical_potential_ti():
    import sympy as sym

    N = 3  # Number of ions
    dim = 3  # Dimension of system
    mass = 171 * cst.m_u  # Mass of an ion

    # Trapping strength (in rad/s)
    omega_x = 2 * np.pi * 0.4e6  # Direction x
    omega_y = 2 * np.pi * 0.36e6  # Direction y
    omega_z = 2 * np.pi * 0.08e6  # Direction z

    # function for the potential defined using autograd.numpy
    def expr(x):
        return np.sum(
            mass * (omega_x) ** 2 / 2 * x[:, 0] ** 2
            + mass * (omega_y) ** 2 / 2 * x[:, 1] ** 2
            + mass * (omega_z) ** 2 / 2 * x[:, 2] ** 2
        )

    # Define an instance of the AutoDiffPotential class for a harmonic potential
    adp = oqd_trical.mechanical.AutoDiffPotential(expr, N=N, dim=dim)

    # Parameters for a Gaussian beam
    focal_point = np.zeros(3)
    beam_waist = 1e-6
    refractive_index = 1.0
    wavelength = 375e-9
    power = 1e0
    Omega_bar = 2.23e6
    transition_wavelength = 369.52e-9

    # Symbol for the directions
    x, y, z = sym.symbols("x, y ,z")

    # Defining intensity expression for the Gaussian beam with the above parameters
    delta_x, delta_y, delta_z = np.array([x, y, z]) - focal_point
    x_R = np.pi * beam_waist**2 * refractive_index / wavelength
    w = beam_waist * sym.sqrt(1 + (delta_x / x_R) ** 2)
    intensity_expr = (
        2
        * power
        / (np.pi * beam_waist**2)
        * (beam_waist / w) ** 2
        * sym.exp(-2 * (delta_y**2 + delta_z**2) / w**2)
    )

    # Define an instance of the SymbolicOpticalPotential class for the Gaussian beam defined above
    sop = oqd_trical.mechanical.SymbolicOpticalPotential(
        intensity_expr,
        wavelength,
        Omega_bar=Omega_bar,
        transition_wavelength=transition_wavelength,
    )

    ti_sop = oqd_trical.mechanical.TrappedIons(N, adp, sop, m=mass)
    return ti_sop


def polynomial_alpha():
    mass = 171 * cst.m_u  # Mass of an ion

    # Trapping strength (in rad/s)
    omega_x = 2 * np.pi * 0.4e6  # Direction x
    omega_y = 2 * np.pi * 0.36e6  # Direction y
    omega_z = 2 * np.pi * 0.08e6  # Direction z

    # Coefficients of the multivariate polynomial consistent with the trapping strength above for a harmonic potential
    alpha = np.zeros((3, 3, 3))
    alpha[2, 0, 0] = mass * (omega_x) ** 2 / 2
    alpha[0, 2, 0] = mass * (omega_y) ** 2 / 2
    alpha[0, 0, 2] = mass * (omega_z) ** 2 / 2
    return alpha


def polynomical_potential_ti():
    N = 3  # Number of ions
    mass = 171 * cst.m_u  # Mass of an ion

    alpha = polynomial_alpha()

    # Harmonic trapped ion system
    pp = oqd_trical.mechanical.PolynomialPotential(alpha, N=N)

    ti = oqd_trical.mechanical.TrappedIons(N, pp, m=mass)
    return ti


def polynomial_fit():
    from oqd_trical.misc.polynomial import multivariate_polyfit

    alpha = polynomial_alpha()
    # Create grid of points
    I_x = np.linspace(-2e-4, 2e-4, 81)
    I_y = np.linspace(-2e-4, 2e-4, 81)
    I_z = np.linspace(-5e-4, 5e-4, 201)
    r = np.array(np.meshgrid(I_x, I_y, I_z))
    r = r.reshape(3, -1).transpose()  # Last axis is axis for the direction x,y and z

    # Function for the potential
    def f(x, y, z):
        return np.polynomial.polynomial.polyval3d(x, y, z, alpha)

    # Values of potential at grid points
    U_r = f(*r.transpose())

    # Parameters of fitting polynomial
    deg = (2, 2, 2)  # degree of polynomial

    alpha_fit = multivariate_polyfit(r, U_r, deg)
    return alpha_fit


def polynomial_potential_fit_ti():
    N = 3  # Number of ions
    mass = 171 * cst.m_u  # Mass of an ion
    alpha_fit = polynomial_fit()
    # Define an instance of the PolynomialPotential class for the coeffiecients obtained from the multivariate polynomial fit
    pfp = oqd_trical.mechanical.PolynomialPotential(alpha_fit, N=N)
    ti_pfp = oqd_trical.mechanical.TrappedIons(
        N, pfp, m=mass
    )  # PolynomialPotential by fitting coeffiencients
    return ti_pfp


class TestMechanical:
    def test_trapped_ions_equilibrium_position_polynomial(self):
        ti = polynomical_potential_ti()

        # Calculate equilibrium position
        x_ep = ti.equilibrium_position()
        assert isinstance(x_ep, np.ndarray)

        reference_x_ep = np.array(
            [
                [0.00000000e00, 0.00000000e00, -1.58999505e-05],
                [-4.46245230e-15, -4.46245230e-15, 1.16541344e-13],
                [0.00000000e00, 0.00000000e00, 1.58999503e-05],
            ]
        )
        np.testing.assert_allclose(x_ep, reference_x_ep, atol=5e-8)

    def test_trapped_ions_normal_modes_polynomial(self):
        ti = polynomical_potential_ti()

        # Calculate equilibrium position
        ti.equilibrium_position()

        # Calculate normal modes
        (w, b) = ti.normal_modes()

        expected_w = np.array(
            [
                2513274.12287183,
                2462495.67600486,
                2389593.76192821,
                2261946.71058465,
                2205389.09473366,
                2123679.46071759,
                1210553.10167286,
                870623.68298652,
                502654.82457437,
            ]
        )
        expected_b = np.array(
            [
                [
                    5.77350269e-01,
                    -7.07106798e-01,
                    4.08248261e-01,
                    1.60441765e-24,
                    -4.32062149e-20,
                    -5.66212304e-20,
                    4.53276538e-11,
                    2.43006547e-11,
                    -3.92951209e-27,
                ],
                [
                    5.77350269e-01,
                    3.40379568e-08,
                    -8.16496581e-01,
                    -1.00848786e-23,
                    -8.20809073e-24,
                    1.13239796e-19,
                    8.06106884e-18,
                    -4.86013089e-11,
                    5.08140183e-25,
                ],
                [
                    5.77350269e-01,
                    7.07106764e-01,
                    4.08248320e-01,
                    2.11920943e-23,
                    4.32281570e-20,
                    -5.66118101e-20,
                    -4.53276619e-11,
                    2.43006542e-11,
                    -2.11129340e-24,
                ],
                [
                    -1.42564002e-23,
                    -3.92972544e-20,
                    5.46872416e-20,
                    5.77350269e-01,
                    7.07106798e-01,
                    4.08248261e-01,
                    6.13353812e-11,
                    3.20734550e-11,
                    8.01759246e-17,
                ],
                [
                    -1.35475580e-24,
                    -3.16545192e-24,
                    -1.09429943e-19,
                    5.77350269e-01,
                    -3.40379538e-08,
                    -8.16496581e-01,
                    -1.21848610e-17,
                    -6.41472022e-11,
                    8.93511628e-17,
                ],
                [
                    -4.26270015e-25,
                    3.92743750e-20,
                    5.47103396e-20,
                    5.77350269e-01,
                    -7.07106764e-01,
                    4.08248320e-01,
                    -6.13355727e-11,
                    3.20735917e-11,
                    -6.98196028e-18,
                ],
                [
                    -3.45037319e-23,
                    2.61699356e-11,
                    -4.20899640e-11,
                    2.18178270e-16,
                    -3.54118941e-11,
                    -5.55529037e-11,
                    4.08248261e-01,
                    7.07106798e-01,
                    5.77350269e-01,
                ],
                [
                    2.88168652e-23,
                    -5.23398709e-11,
                    -8.21679738e-18,
                    -1.20779618e-16,
                    7.08240499e-11,
                    -4.63810603e-17,
                    -8.16496581e-01,
                    -3.40379518e-08,
                    5.77350269e-01,
                ],
                [
                    5.09733099e-24,
                    2.61699353e-11,
                    4.20899723e-11,
                    1.23495432e-16,
                    -3.54120850e-11,
                    5.55529679e-11,
                    4.08248320e-01,
                    -7.07106764e-01,
                    5.77350269e-01,
                ],
            ]
        )

        np.testing.assert_allclose(w, expected_w)
        assert_eigenvectors_equal_up_to_phase(b, expected_b)
        # Calculate normal modes in principal axes if it exist
        (x_pa, w_pa, b_pa) = ti.principal_axes()

        expected_x_pa = np.eye(3)
        expected_w_pa = np.array(
            [
                2513274.12287183,
                2462495.67600486,
                2389593.76192821,
                2261946.71058465,
                2205389.09473366,
                2123679.46071759,
                1210553.10167286,
                870623.68298652,
                502654.82457437,
            ]
        )
        expected_b_pa = np.array(
            [
                [
                    5.77350269e-01,
                    -7.07106798e-01,
                    4.08248261e-01,
                    7.73802594e-20,
                    5.15628979e-20,
                    -1.90626326e-21,
                    4.53276579e-11,
                    2.43006618e-11,
                    5.81015314e-18,
                ],
                [
                    5.77350269e-01,
                    3.40379568e-08,
                    -8.16496581e-01,
                    7.73685701e-20,
                    -8.21193988e-24,
                    3.80985248e-21,
                    -1.55728540e-19,
                    -4.86013089e-11,
                    5.81015365e-18,
                ],
                [
                    5.77350269e-01,
                    7.07106764e-01,
                    4.08248320e-01,
                    7.73998471e-20,
                    -5.15409520e-20,
                    -1.89683395e-21,
                    -4.53276578e-11,
                    2.43006471e-11,
                    5.81015103e-18,
                ],
                [
                    -8.00868839e-20,
                    5.87712894e-20,
                    -1.93265462e-21,
                    5.77350269e-01,
                    7.07106798e-01,
                    4.08248261e-01,
                    6.13354044e-11,
                    3.20734952e-11,
                    1.12972287e-16,
                ],
                [
                    -8.00739823e-20,
                    -3.17314581e-24,
                    3.80985248e-21,
                    5.77350269e-01,
                    -3.40379538e-08,
                    -8.16496581e-01,
                    -5.85659213e-17,
                    -6.41472022e-11,
                    1.22147525e-16,
                ],
                [
                    -8.00730538e-20,
                    -5.87941611e-20,
                    -1.90955998e-21,
                    5.77350269e-01,
                    -7.07106764e-01,
                    4.08248320e-01,
                    -6.13355495e-11,
                    3.20735515e-11,
                    2.58144020e-17,
                ],
                [
                    -5.70007094e-18,
                    2.61699425e-11,
                    -4.20899681e-11,
                    2.26794268e-16,
                    -3.54118836e-11,
                    -5.55528976e-11,
                    4.08248261e-01,
                    7.07106798e-01,
                    5.77350269e-01,
                ],
                [
                    -5.70000762e-18,
                    -5.23398709e-11,
                    -1.55728540e-19,
                    -1.12163620e-16,
                    7.08240499e-11,
                    -5.85659213e-17,
                    -8.16496581e-01,
                    -3.40379518e-08,
                    5.77350269e-01,
                ],
                [
                    -5.70003134e-18,
                    2.61699284e-11,
                    4.20899682e-11,
                    1.32111430e-16,
                    -3.54120956e-11,
                    5.55529740e-11,
                    4.08248320e-01,
                    -7.07106764e-01,
                    5.77350269e-01,
                ],
            ]
        )

        np.testing.assert_allclose(x_pa, expected_x_pa)
        np.testing.assert_allclose(w_pa, expected_w_pa)
        assert_eigenvectors_equal_up_to_phase(b_pa, expected_b_pa)

    def test_trapped_ions_normal_modes_gaussian(self):
        ti = gaussian_optical_potential_ti()

        # Calculate equilibrium position
        ti.equilibrium_position()

        # Calculate normal modes
        (w, b) = ti.normal_modes()

        expected_w = np.array(
            [
                4527812.814222,
                4106707.730764,
                2513274.122872,
                2462495.668243,
                2389593.742731,
                2215633.312998,
                2205389.086067,
                870623.726896,
                797969.334531,
            ]
        )

        np.testing.assert_allclose(w, expected_w)
        # Calculate normal modes in principal axes if it exist
        (x_pa, w_pa, b_pa) = ti.principal_axes()

        expected_x_pa = np.eye(3)
        expected_w_pa = np.array(
            [
                2513274.122872,
                2462495.668243,
                2389593.742731,
                4527812.814222,
                2215633.312998,
                2205389.086067,
                4106707.730764,
                870623.726896,
                797969.334531,
            ]
        )

        np.testing.assert_allclose(x_pa, expected_x_pa)
        np.testing.assert_allclose(w_pa, expected_w_pa)

    def test_trapped_ions_normal_modes_symbolic(self):
        ti = symbolic_optical_potential_ti()

        # Calculate equilibrium position
        ti.equilibrium_position()

        # Calculate normal modes
        (w, b) = ti.normal_modes()

        expected_w = np.array(
            [
                4527812.822414,
                4106707.711302,
                2521690.973055,
                2462495.67791,
                2404209.895143,
                2215633.322042,
                2205389.096861,
                870623.672211,
                797969.291497,
            ]
        )

        np.testing.assert_allclose(w, expected_w)
        # Calculate normal modes in principal axes if it exist
        (x_pa, w_pa, b_pa) = ti.principal_axes()

        expected_x_pa = np.eye(3)
        expected_w_pa = np.array(
            [
                2521690.973055,
                2462495.67791,
                2404209.895143,
                4527812.822414,
                2215633.322042,
                2205389.096861,
                4106707.711302,
                870623.672211,
                797969.291497,
            ]
        )

        np.testing.assert_allclose(x_pa, expected_x_pa)
        np.testing.assert_allclose(w_pa, expected_w_pa)

    def test_trapped_ions_normal_modes_ms(self):
        ti = ms_potential_ti()

        # Calculate equilibrium position
        ti.equilibrium_position()

        # Calculate normal modes
        (w, b) = ti.normal_modes()

        expected_w = np.array(
            [
                3038136.228329,
                2719526.418053,
                2466155.182154,
                2462495.67687,
                2207593.999445,
                2205389.095699,
                1304741.642408,
                870623.678094,
                519143.816445,
            ]
        )

        np.testing.assert_allclose(w, expected_w, rtol=1e-6)
        # Calculate normal modes in principal axes if it exist
        (x_pa, w_pa, b_pa) = ti.principal_axes()

        expected_x_pa = np.eye(3)
        expected_w_pa = np.array(
            [
                3038136.228329,
                2466155.182154,
                2462495.67687,
                2719526.418053,
                2207593.999445,
                2205389.095699,
                1304741.642408,
                870623.678094,
                519143.816445,
            ]
        )

        np.testing.assert_allclose(x_pa, expected_x_pa)
        np.testing.assert_allclose(w_pa, expected_w_pa, rtol=1e-6)

    def test_trapped_ions_optical_potential(self):
        ti = optical_potential_ti()

        # Calculate equilibrium position
        ti.equilibrium_position()

        # Calculate normal modes
        (w, b) = ti.normal_modes()

        expected_w = np.array(
            [
                4527812.822414,
                4106707.711303,
                2521690.973055,
                2462495.67791,
                2404209.895143,
                2215633.322042,
                2205389.096861,
                870623.672211,
                797969.291497,
            ]
        )

        np.testing.assert_allclose(w, expected_w)
        # Calculate normal modes in principal axes if it exist
        (x_pa, w_pa, b_pa) = ti.principal_axes()

        expected_x_pa = np.eye(3)
        expected_w_pa = np.array(
            [
                2521690.973055,
                2462495.67791,
                2404209.895143,
                4527812.822414,
                2215633.322042,
                2205389.096861,
                4106707.711303,
                870623.672211,
                797969.291497,
            ]
        )

        np.testing.assert_allclose(x_pa, expected_x_pa)
        np.testing.assert_allclose(w_pa, expected_w_pa)

    def test_trapped_ions_polynomial_potential_fit(self):
        ti = polynomial_potential_fit_ti()

        hess = [p.hessian()(np.eye(3)) for p in ti.ps][0]
        expected_hess = np.array(
            [
                [
                    1.79359724e-12,
                    0.00000000e00,
                    0.00000000e00,
                    2.93684246e-25,
                    0.00000000e00,
                    0.00000000e00,
                    6.24323677e-26,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    1.79359724e-12,
                    0.00000000e00,
                    0.00000000e00,
                    1.84652814e-25,
                    0.00000000e00,
                    0.00000000e00,
                    -4.60910113e-22,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    1.79359724e-12,
                    0.00000000e00,
                    0.00000000e00,
                    8.47898991e-22,
                    0.00000000e00,
                    0.00000000e00,
                    3.45998010e-26,
                ],
                [
                    2.93684246e-25,
                    0.00000000e00,
                    0.00000000e00,
                    1.45281376e-12,
                    0.00000000e00,
                    0.00000000e00,
                    -4.87994715e-22,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    1.84652814e-25,
                    0.00000000e00,
                    0.00000000e00,
                    1.45281377e-12,
                    0.00000000e00,
                    0.00000000e00,
                    1.38217702e-25,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    8.47898991e-22,
                    0.00000000e00,
                    0.00000000e00,
                    1.45281377e-12,
                    0.00000000e00,
                    0.00000000e00,
                    1.76963699e-24,
                ],
                [
                    6.24323677e-26,
                    0.00000000e00,
                    0.00000000e00,
                    -4.87994715e-22,
                    0.00000000e00,
                    0.00000000e00,
                    7.17438891e-14,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    -4.60910113e-22,
                    0.00000000e00,
                    0.00000000e00,
                    1.38217702e-25,
                    0.00000000e00,
                    0.00000000e00,
                    7.17438893e-14,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    3.45998010e-26,
                    0.00000000e00,
                    0.00000000e00,
                    1.76963699e-24,
                    0.00000000e00,
                    0.00000000e00,
                    7.17438897e-14,
                ],
            ]
        )

        np.testing.assert_allclose(hess, expected_hess, atol=1e-12)
        # Calculate equilibrium position
        ti.equilibrium_position()

        # Calculate normal modes
        (w, b) = ti.normal_modes()

        expected_w = np.array(
            [
                2513274.122872,
                2462495.668113,
                2389593.742409,
                2261946.710585,
                2205389.085922,
                2123679.438755,
                1210553.178732,
                870623.727631,
                502654.824575,
            ]
        )

        np.testing.assert_allclose(w, expected_w)
        # Calculate normal modes in principal axes if it exist
        (x_pa, w_pa, b_pa) = ti.principal_axes()

        expected_x_pa = np.eye(3)
        expected_w_pa = np.array(
            [
                2513274.122872,
                2462495.668113,
                2389593.742409,
                2261946.710585,
                2205389.085922,
                2123679.438755,
                1210553.178732,
                870623.727631,
                502654.824575,
            ]
        )

        np.testing.assert_allclose(x_pa, expected_x_pa, atol=5e-8)
        np.testing.assert_allclose(w_pa, expected_w_pa)

    def test_trapped_ions_afvanced_symbolic_potential(self):
        ti = advanced_symbolic_potential_ti()

        # Calculate equilibrium position
        ti.equilibrium_position()

        # Calculate normal modes
        (w, b) = ti.normal_modes()

        expected_w = np.array(
            [
                2513274.122872,
                2462495.677579,
                2389593.765823,
                2261946.710585,
                2205389.096492,
                2123679.4651,
                1210553.086298,
                870623.674079,
                502654.824574,
            ]
        )

        np.testing.assert_allclose(w, expected_w, rtol=1e-6)
        # Calculate normal modes in principal axes if it exist
        (x_pa, w_pa, b_pa) = ti.principal_axes()

        expected_x_pa = np.eye(3)
        expected_w_pa = np.array(
            [
                2513274.122872,
                2462495.677579,
                2389593.765823,
                2261946.710585,
                2205389.096492,
                2123679.4651,
                1210553.086298,
                870623.674079,
                502654.824574,
            ]
        )

        np.testing.assert_allclose(x_pa, expected_x_pa)
        np.testing.assert_allclose(w_pa, expected_w_pa, rtol=1e-6)

    def test_mod_ion_coupling(self):
        ti = polynomical_potential_ti()

        mic = ti.mode_ion_coupling()

        expected_mic = np.array(
            [
                [
                    [
                        0.00000000e00,
                        3.33333333e-01,
                        3.33333333e-01,
                        -4.62381840e-20,
                        -4.62307352e-20,
                        -4.62301992e-20,
                        -3.29093749e-18,
                        -3.29090093e-18,
                        -3.29091463e-18,
                    ],
                    [
                        3.33333333e-01,
                        0.00000000e00,
                        3.33333333e-01,
                        -4.62381840e-20,
                        -4.62307352e-20,
                        -4.62301992e-20,
                        -3.29093749e-18,
                        -3.29090093e-18,
                        -3.29091463e-18,
                    ],
                    [
                        3.33333333e-01,
                        3.33333333e-01,
                        0.00000000e00,
                        -4.62381840e-20,
                        -4.62307352e-20,
                        -4.62301992e-20,
                        -3.29093749e-18,
                        -3.29090093e-18,
                        -3.29091463e-18,
                    ],
                    [
                        -4.62381840e-20,
                        -4.62381840e-20,
                        -4.62381840e-20,
                        0.00000000e00,
                        6.41287573e-39,
                        6.41280137e-39,
                        4.56500920e-37,
                        4.56495849e-37,
                        4.56497748e-37,
                    ],
                    [
                        -4.62307352e-20,
                        -4.62307352e-20,
                        -4.62307352e-20,
                        6.41287573e-39,
                        0.00000000e00,
                        6.41176829e-39,
                        4.56427380e-37,
                        4.56422309e-37,
                        4.56424209e-37,
                    ],
                    [
                        -4.62301992e-20,
                        -4.62301992e-20,
                        -4.62301992e-20,
                        6.41280137e-39,
                        6.41176829e-39,
                        0.00000000e00,
                        4.56422087e-37,
                        4.56417017e-37,
                        4.56418916e-37,
                    ],
                    [
                        -3.29093749e-18,
                        -3.29093749e-18,
                        -3.29093749e-18,
                        4.56500920e-37,
                        4.56427380e-37,
                        4.56422087e-37,
                        0.00000000e00,
                        3.24904478e-35,
                        3.24905830e-35,
                    ],
                    [
                        -3.29090093e-18,
                        -3.29090093e-18,
                        -3.29090093e-18,
                        4.56495849e-37,
                        4.56422309e-37,
                        4.56417017e-37,
                        3.24904478e-35,
                        0.00000000e00,
                        3.24902221e-35,
                    ],
                    [
                        -3.29091463e-18,
                        -3.29091463e-18,
                        -3.29091463e-18,
                        4.56497748e-37,
                        4.56424209e-37,
                        4.56418916e-37,
                        3.24905830e-35,
                        3.24902221e-35,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        -2.40684706e-08,
                        -5.00000000e-01,
                        -4.15575783e-20,
                        2.24375297e-24,
                        4.15737510e-20,
                        -1.85049443e-11,
                        3.70098785e-11,
                        -1.85049343e-11,
                    ],
                    [
                        -2.40684706e-08,
                        0.00000000e00,
                        2.40684695e-08,
                        2.00045461e-27,
                        -1.08007400e-31,
                        -2.00123311e-27,
                        8.90771373e-19,
                        -1.78154226e-18,
                        8.90770890e-19,
                    ],
                    [
                        -5.00000000e-01,
                        2.40684695e-08,
                        0.00000000e00,
                        4.15575763e-20,
                        -2.24375286e-24,
                        -4.15737490e-20,
                        1.85049434e-11,
                        -3.70098768e-11,
                        1.85049334e-11,
                    ],
                    [
                        -4.15575783e-20,
                        2.00045461e-27,
                        4.15575763e-20,
                        0.00000000e00,
                        -1.86489871e-43,
                        -3.45540866e-39,
                        1.53804127e-30,
                        -3.07608170e-30,
                        1.53804043e-30,
                    ],
                    [
                        2.24375297e-24,
                        -1.08007400e-31,
                        -2.24375286e-24,
                        -1.86489871e-43,
                        0.00000000e00,
                        1.86562446e-43,
                        -8.30410435e-35,
                        1.66082042e-34,
                        -8.30409985e-35,
                    ],
                    [
                        4.15737510e-20,
                        -2.00123311e-27,
                        -4.15737490e-20,
                        -3.45540866e-39,
                        1.86562446e-43,
                        0.00000000e00,
                        -1.53863982e-30,
                        3.07727880e-30,
                        -1.53863898e-30,
                    ],
                    [
                        -1.85049443e-11,
                        8.90771373e-19,
                        1.85049434e-11,
                        1.53804127e-30,
                        -8.30410435e-35,
                        -1.53863982e-30,
                        0.00000000e00,
                        -1.36973141e-21,
                        6.84865522e-22,
                    ],
                    [
                        3.70098785e-11,
                        -1.78154226e-18,
                        -3.70098768e-11,
                        -3.07608170e-30,
                        1.66082042e-34,
                        3.07727880e-30,
                        -1.36973141e-21,
                        0.00000000e00,
                        -1.36973067e-21,
                    ],
                    [
                        -1.85049343e-11,
                        8.90770890e-19,
                        1.85049334e-11,
                        1.53804043e-30,
                        -8.30409985e-35,
                        -1.53863898e-30,
                        6.84865522e-22,
                        -1.36973067e-21,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        -3.33333309e-01,
                        1.66666667e-01,
                        -7.89002889e-22,
                        1.55536565e-21,
                        -7.79574543e-22,
                        -1.71831563e-11,
                        -6.35759054e-20,
                        1.71831563e-11,
                    ],
                    [
                        -3.33333309e-01,
                        0.00000000e00,
                        -3.33333357e-01,
                        1.57800589e-21,
                        -3.11073153e-21,
                        1.55914920e-21,
                        3.43663150e-11,
                        1.27151820e-19,
                        -3.43663151e-11,
                    ],
                    [
                        1.66666667e-01,
                        -3.33333357e-01,
                        0.00000000e00,
                        -7.89003003e-22,
                        1.55536588e-21,
                        -7.79574655e-22,
                        -1.71831588e-11,
                        -6.35759146e-20,
                        1.71831588e-11,
                    ],
                    [
                        -7.89002889e-22,
                        1.57800589e-21,
                        -7.89003003e-22,
                        0.00000000e00,
                        -7.36312902e-42,
                        3.69051993e-42,
                        8.13453714e-32,
                        3.00969482e-40,
                        -8.13453717e-32,
                    ],
                    [
                        1.55536565e-21,
                        -3.11073153e-21,
                        1.55536588e-21,
                        -7.36312902e-42,
                        0.00000000e00,
                        -7.27514184e-42,
                        -1.60356569e-31,
                        -5.93302763e-40,
                        1.60356570e-31,
                    ],
                    [
                        -7.79574543e-22,
                        1.55914920e-21,
                        -7.79574655e-22,
                        3.69051993e-42,
                        -7.27514184e-42,
                        0.00000000e00,
                        8.03733187e-32,
                        2.97372987e-40,
                        -8.03733190e-32,
                    ],
                    [
                        -1.71831563e-11,
                        3.43663150e-11,
                        -1.71831588e-11,
                        8.13453714e-32,
                        -1.60356569e-31,
                        8.03733187e-32,
                        0.00000000e00,
                        6.55460926e-30,
                        -1.77156542e-21,
                    ],
                    [
                        -6.35759054e-20,
                        1.27151820e-19,
                        -6.35759146e-20,
                        3.00969482e-40,
                        -5.93302763e-40,
                        2.97372987e-40,
                        6.55460926e-30,
                        0.00000000e00,
                        -6.55460928e-30,
                    ],
                    [
                        1.71831563e-11,
                        -3.43663151e-11,
                        1.71831588e-11,
                        -8.13453717e-32,
                        1.60356570e-31,
                        -8.03733190e-32,
                        -1.77156542e-21,
                        -6.55460928e-30,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        5.98680003e-39,
                        5.98922025e-39,
                        4.46755136e-20,
                        4.46755136e-20,
                        4.46755136e-20,
                        1.75493993e-35,
                        -8.67925001e-36,
                        1.02228167e-35,
                    ],
                    [
                        5.98680003e-39,
                        0.00000000e00,
                        5.98831550e-39,
                        4.46687648e-20,
                        4.46687648e-20,
                        4.46687648e-20,
                        1.75467482e-35,
                        -8.67793890e-36,
                        1.02212724e-35,
                    ],
                    [
                        5.98922025e-39,
                        5.98831550e-39,
                        0.00000000e00,
                        4.46868226e-20,
                        4.46868226e-20,
                        4.46868226e-20,
                        1.75538417e-35,
                        -8.68144703e-36,
                        1.02254044e-35,
                    ],
                    [
                        4.46755136e-20,
                        4.46687648e-20,
                        4.46868226e-20,
                        0.00000000e00,
                        3.33333333e-01,
                        3.33333333e-01,
                        1.30939732e-16,
                        -6.47576962e-17,
                        7.62745694e-17,
                    ],
                    [
                        4.46755136e-20,
                        4.46687648e-20,
                        4.46868226e-20,
                        3.33333333e-01,
                        0.00000000e00,
                        3.33333333e-01,
                        1.30939732e-16,
                        -6.47576962e-17,
                        7.62745694e-17,
                    ],
                    [
                        4.46755136e-20,
                        4.46687648e-20,
                        4.46868226e-20,
                        3.33333333e-01,
                        3.33333333e-01,
                        0.00000000e00,
                        1.30939732e-16,
                        -6.47576962e-17,
                        7.62745694e-17,
                    ],
                    [
                        1.75493993e-35,
                        1.75467482e-35,
                        1.75538417e-35,
                        1.30939732e-16,
                        1.30939732e-16,
                        1.30939732e-16,
                        0.00000000e00,
                        -2.54380661e-32,
                        2.99621149e-32,
                    ],
                    [
                        -8.67925001e-36,
                        -8.67793890e-36,
                        -8.68144703e-36,
                        -6.47576962e-17,
                        -6.47576962e-17,
                        -6.47576962e-17,
                        -2.54380661e-32,
                        0.00000000e00,
                        -1.48180962e-32,
                    ],
                    [
                        1.02228167e-35,
                        1.02212724e-35,
                        1.02254044e-35,
                        7.62745694e-17,
                        7.62745694e-17,
                        7.62745694e-17,
                        2.99621149e-32,
                        -1.48180962e-32,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        -4.23431418e-43,
                        -2.65760084e-39,
                        3.64604756e-20,
                        -1.75509554e-27,
                        -3.64604739e-20,
                        -1.82593934e-30,
                        3.65189325e-30,
                        -1.82595027e-30,
                    ],
                    [
                        -4.23431418e-43,
                        0.00000000e00,
                        4.23251199e-43,
                        -5.80671852e-24,
                        2.79517630e-31,
                        5.80671824e-24,
                        2.90800259e-34,
                        -5.81602841e-34,
                        2.90802000e-34,
                    ],
                    [
                        -2.65760084e-39,
                        4.23251199e-43,
                        0.00000000e00,
                        -3.64449575e-20,
                        1.75434854e-27,
                        3.64449558e-20,
                        1.82516219e-30,
                        -3.65033896e-30,
                        1.82517312e-30,
                    ],
                    [
                        3.64604756e-20,
                        -5.80671852e-24,
                        -3.64449575e-20,
                        0.00000000e00,
                        -2.40684685e-08,
                        -5.00000000e-01,
                        -2.50399836e-11,
                        5.00801672e-11,
                        -2.50401335e-11,
                    ],
                    [
                        -1.75509554e-27,
                        2.79517630e-31,
                        1.75434854e-27,
                        -2.40684685e-08,
                        0.00000000e00,
                        2.40684674e-08,
                        1.20534806e-18,
                        -2.41070574e-18,
                        1.20535527e-18,
                    ],
                    [
                        -3.64604739e-20,
                        5.80671824e-24,
                        3.64449558e-20,
                        -5.00000000e-01,
                        2.40684674e-08,
                        0.00000000e00,
                        2.50399824e-11,
                        -5.00801648e-11,
                        2.50401323e-11,
                    ],
                    [
                        -1.82593934e-30,
                        2.90800259e-34,
                        1.82516219e-30,
                        -2.50399836e-11,
                        1.20534806e-18,
                        2.50399824e-11,
                        0.00000000e00,
                        -2.50801301e-21,
                        1.25400901e-21,
                    ],
                    [
                        3.65189325e-30,
                        -5.81602841e-34,
                        -3.65033896e-30,
                        5.00801672e-11,
                        -2.41070574e-18,
                        -5.00801648e-11,
                        -2.50801301e-21,
                        0.00000000e00,
                        -2.50802803e-21,
                    ],
                    [
                        -1.82595027e-30,
                        2.90802000e-34,
                        1.82517312e-30,
                        -2.50401335e-11,
                        1.20535527e-18,
                        2.50401323e-11,
                        1.25400901e-21,
                        -2.50802803e-21,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        -7.26258181e-42,
                        3.61586487e-42,
                        -7.78228661e-22,
                        1.55645743e-21,
                        -7.78228773e-22,
                        1.05898448e-31,
                        1.11642064e-37,
                        -1.05898593e-31,
                    ],
                    [
                        -7.26258181e-42,
                        0.00000000e00,
                        -7.22665753e-42,
                        1.55536565e-21,
                        -3.11073153e-21,
                        1.55536588e-21,
                        -2.11648345e-31,
                        -2.23127521e-37,
                        2.11648636e-31,
                    ],
                    [
                        3.61586487e-42,
                        -7.22665753e-42,
                        0.00000000e00,
                        -7.74379161e-22,
                        1.54875843e-21,
                        -7.74379273e-22,
                        1.05374622e-31,
                        1.11089828e-37,
                        -1.05374767e-31,
                    ],
                    [
                        -7.78228661e-22,
                        1.55536565e-21,
                        -7.74379161e-22,
                        0.00000000e00,
                        -3.33333309e-01,
                        1.66666667e-01,
                        -2.26793738e-11,
                        -2.39094355e-17,
                        2.26794050e-11,
                    ],
                    [
                        1.55645743e-21,
                        -3.11073153e-21,
                        1.54875843e-21,
                        -3.33333309e-01,
                        0.00000000e00,
                        -3.33333357e-01,
                        4.53587509e-11,
                        4.78188745e-17,
                        -4.53588133e-11,
                    ],
                    [
                        -7.78228773e-22,
                        1.55536588e-21,
                        -7.74379273e-22,
                        1.66666667e-01,
                        -3.33333357e-01,
                        0.00000000e00,
                        -2.26793771e-11,
                        -2.39094390e-17,
                        2.26794083e-11,
                    ],
                    [
                        1.05898448e-31,
                        -2.11648345e-31,
                        1.05374622e-31,
                        -2.26793738e-11,
                        4.53587509e-11,
                        -2.26793771e-11,
                        0.00000000e00,
                        3.25350663e-27,
                        -3.08612867e-21,
                    ],
                    [
                        1.11642064e-37,
                        -2.23127521e-37,
                        1.11089828e-37,
                        -2.39094355e-17,
                        4.78188745e-17,
                        -2.39094390e-17,
                        3.25350663e-27,
                        0.00000000e00,
                        -3.25351110e-27,
                    ],
                    [
                        -1.05898593e-31,
                        2.11648636e-31,
                        -1.05374767e-31,
                        2.26794050e-11,
                        -4.53588133e-11,
                        2.26794083e-11,
                        -3.08612867e-21,
                        -3.25351110e-27,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        -7.05880997e-30,
                        -2.05459656e-21,
                        2.78019023e-21,
                        -2.65465605e-27,
                        -2.78019681e-21,
                        1.85049375e-11,
                        -3.70098777e-11,
                        1.85049402e-11,
                    ],
                    [
                        -7.05880997e-30,
                        0.00000000e00,
                        7.05880994e-30,
                        -9.55167295e-30,
                        9.12038539e-36,
                        9.55169555e-30,
                        -6.35759054e-20,
                        1.27151820e-19,
                        -6.35759146e-20,
                    ],
                    [
                        -2.05459656e-21,
                        7.05880994e-30,
                        0.00000000e00,
                        -2.78019022e-21,
                        2.65465604e-27,
                        2.78019680e-21,
                        -1.85049375e-11,
                        3.70098776e-11,
                        -1.85049401e-11,
                    ],
                    [
                        2.78019023e-21,
                        -9.55167295e-30,
                        -2.78019022e-21,
                        0.00000000e00,
                        -3.59216447e-27,
                        -3.76204074e-21,
                        2.50400722e-11,
                        -5.00801480e-11,
                        2.50400758e-11,
                    ],
                    [
                        -2.65465605e-27,
                        9.12038539e-36,
                        2.65465604e-27,
                        -3.59216447e-27,
                        0.00000000e00,
                        3.59217297e-27,
                        -2.39094355e-17,
                        4.78188745e-17,
                        -2.39094390e-17,
                    ],
                    [
                        -2.78019681e-21,
                        9.55169555e-30,
                        2.78019680e-21,
                        -3.76204074e-21,
                        3.59217297e-27,
                        0.00000000e00,
                        -2.50401314e-11,
                        5.00802665e-11,
                        -2.50401351e-11,
                    ],
                    [
                        1.85049375e-11,
                        -6.35759054e-20,
                        -1.85049375e-11,
                        2.50400722e-11,
                        -2.39094355e-17,
                        -2.50401314e-11,
                        0.00000000e00,
                        -3.33333309e-01,
                        1.66666667e-01,
                    ],
                    [
                        -3.70098777e-11,
                        1.27151820e-19,
                        3.70098776e-11,
                        -5.00801480e-11,
                        4.78188745e-17,
                        5.00802665e-11,
                        -3.33333309e-01,
                        0.00000000e00,
                        -3.33333357e-01,
                    ],
                    [
                        1.85049402e-11,
                        -6.35759146e-20,
                        -1.85049401e-11,
                        2.50400758e-11,
                        -2.39094390e-17,
                        -2.50401351e-11,
                        1.66666667e-01,
                        -3.33333357e-01,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        -1.18104397e-21,
                        5.90521806e-22,
                        7.79407159e-22,
                        -1.55881947e-21,
                        7.79408528e-22,
                        1.71831632e-11,
                        -8.27144755e-19,
                        -1.71831623e-11,
                    ],
                    [
                        -1.18104397e-21,
                        0.00000000e00,
                        -1.18104325e-21,
                        -1.55881385e-21,
                        3.11763799e-21,
                        -1.55881658e-21,
                        -3.43663159e-11,
                        1.65428901e-18,
                        3.43663143e-11,
                    ],
                    [
                        5.90521806e-22,
                        -1.18104325e-21,
                        0.00000000e00,
                        7.79406687e-22,
                        -1.55881852e-21,
                        7.79408055e-22,
                        1.71831528e-11,
                        -8.27144254e-19,
                        -1.71831519e-11,
                    ],
                    [
                        7.79407159e-22,
                        -1.55881385e-21,
                        7.79406687e-22,
                        0.00000000e00,
                        -2.05742498e-21,
                        1.02871090e-21,
                        2.26793865e-11,
                        -1.09171608e-18,
                        -2.26793854e-11,
                    ],
                    [
                        -1.55881947e-21,
                        3.11763799e-21,
                        -1.55881852e-21,
                        -2.05742498e-21,
                        0.00000000e00,
                        -2.05742859e-21,
                        -4.53589228e-11,
                        2.18343938e-18,
                        4.53589206e-11,
                    ],
                    [
                        7.79408528e-22,
                        -1.55881658e-21,
                        7.79408055e-22,
                        1.02871090e-21,
                        -2.05742859e-21,
                        0.00000000e00,
                        2.26794263e-11,
                        -1.09171800e-18,
                        -2.26794252e-11,
                    ],
                    [
                        1.71831632e-11,
                        -3.43663159e-11,
                        1.71831528e-11,
                        2.26793865e-11,
                        -4.53589228e-11,
                        2.26794263e-11,
                        0.00000000e00,
                        -2.40684671e-08,
                        -5.00000000e-01,
                    ],
                    [
                        -8.27144755e-19,
                        1.65428901e-18,
                        -8.27144254e-19,
                        -1.09171608e-18,
                        2.18343938e-18,
                        -1.09171800e-18,
                        -2.40684671e-08,
                        0.00000000e00,
                        2.40684660e-08,
                    ],
                    [
                        -1.71831623e-11,
                        3.43663143e-11,
                        -1.71831519e-11,
                        -2.26793854e-11,
                        4.53589206e-11,
                        -2.26794252e-11,
                        -5.00000000e-01,
                        2.40684660e-08,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        3.37578825e-35,
                        3.37578673e-35,
                        6.56386287e-34,
                        7.09695826e-34,
                        1.49985629e-34,
                        3.35449348e-18,
                        3.35449348e-18,
                        3.35449348e-18,
                    ],
                    [
                        3.37578825e-35,
                        0.00000000e00,
                        3.37578702e-35,
                        6.56386345e-34,
                        7.09695889e-34,
                        1.49985642e-34,
                        3.35449378e-18,
                        3.35449378e-18,
                        3.35449378e-18,
                    ],
                    [
                        3.37578673e-35,
                        3.37578702e-35,
                        0.00000000e00,
                        6.56386049e-34,
                        7.09695569e-34,
                        1.49985574e-34,
                        3.35449226e-18,
                        3.35449226e-18,
                        3.35449226e-18,
                    ],
                    [
                        6.56386287e-34,
                        6.56386345e-34,
                        6.56386049e-34,
                        0.00000000e00,
                        1.37992852e-32,
                        2.91631203e-33,
                        6.52245802e-17,
                        6.52245802e-17,
                        6.52245802e-17,
                    ],
                    [
                        7.09695826e-34,
                        7.09695889e-34,
                        7.09695569e-34,
                        1.37992852e-32,
                        0.00000000e00,
                        3.15316531e-33,
                        7.05219065e-17,
                        7.05219065e-17,
                        7.05219065e-17,
                    ],
                    [
                        1.49985629e-34,
                        1.49985642e-34,
                        1.49985574e-34,
                        2.91631203e-33,
                        3.15316531e-33,
                        0.00000000e00,
                        1.49039519e-17,
                        1.49039519e-17,
                        1.49039519e-17,
                    ],
                    [
                        3.35449348e-18,
                        3.35449378e-18,
                        3.35449226e-18,
                        6.52245802e-17,
                        7.05219065e-17,
                        1.49039519e-17,
                        0.00000000e00,
                        3.33333333e-01,
                        3.33333333e-01,
                    ],
                    [
                        3.35449348e-18,
                        3.35449378e-18,
                        3.35449226e-18,
                        6.52245802e-17,
                        7.05219065e-17,
                        1.49039519e-17,
                        3.33333333e-01,
                        0.00000000e00,
                        3.33333333e-01,
                    ],
                    [
                        3.35449348e-18,
                        3.35449378e-18,
                        3.35449226e-18,
                        6.52245802e-17,
                        7.05219065e-17,
                        1.49039519e-17,
                        3.33333333e-01,
                        3.33333333e-01,
                        0.00000000e00,
                    ],
                ],
            ]
        )
        np.testing.assert_allclose(mic, expected_mic, atol=5e-8)

    def test_polyval(self):
        alpha = polynomial_alpha()
        alpha_fit = polynomial_fit()
        np.testing.assert_allclose(alpha, alpha_fit, atol=1e-8)

        val = polyval(jnp.array([1, 2, 3]), alpha_fit)

        np.testing.assert_allclose(val, np.array([4.125274e-12]), rtol=1e-6)
