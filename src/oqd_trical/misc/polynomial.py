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

"""
Module containing relevant functions regarding polynomials for TrIcal.
"""

from collections.abc import Callable, Sequence
from functools import partial

########################################################################################
import jax
import numpy as np
from jax import numpy as jnp
from jax.typing import ArrayLike

########################################################################################
from .optimize import dflt_ls_opt

########################################################################################


PolynomialFitSolver = Callable[[ArrayLike, ArrayLike], jax.Array]
PolynomialFitSolverFactory = Callable[[Sequence[int]], PolynomialFitSolver]


@jax.jit
def polyval1d(x: ArrayLike, c: ArrayLike) -> jax.Array:
    """
    Evaluate a 1-D polynomial with low-to-high coefficient ordering.

    JAX analog to numpy.polynomial.polynomial.polyval.
    """
    return jnp.polyval(jnp.asarray(c)[::-1], jnp.asarray(x))


@jax.jit
def polyval2d(x: ArrayLike, y: ArrayLike, c: ArrayLike) -> jax.Array:
    """
    Evaluate a 2-D polynomial at points (x, y).
    JAX analog to numpy.polynomial.polynomial.polyval2d.
    """
    c = jnp.asarray(c)
    if c.ndim < 2:
        c = jnp.expand_dims(c, axis=-1)

    x = jnp.asarray(x)
    y = jnp.asarray(y)

    c_y = c[:, ::-1]

    eval_y = jax.vmap(jnp.polyval, in_axes=(0, None))(c_y, y)
    eval_x = eval_y[::-1]

    return jnp.polyval(eval_x, x)


@jax.jit
def polyval3d(x: ArrayLike, y: ArrayLike, z: ArrayLike, c: ArrayLike) -> jax.Array:
    """
    Evaluate a 3-D polynomial at points (x, y, z).
    JAX analog to numpy.polynomial.polynomial.polyval3d.
    """
    c = jnp.asarray(c)
    while c.ndim < 3:
        c = jnp.expand_dims(c, axis=-1)

    x = jnp.asarray(x)
    y = jnp.asarray(y)
    z = jnp.asarray(z)

    c_z = c[:, :, ::-1]
    eval_z = jax.vmap(jax.vmap(jnp.polyval, in_axes=(0, None)), in_axes=(0, None))(
        c_z, z
    )

    c_y = eval_z[:, ::-1]
    eval_y = jax.vmap(jnp.polyval, in_axes=(0, None))(c_y, y)

    c_x = eval_y[::-1]

    return jnp.polyval(c_x, x)


@partial(jax.jit, static_argnames=["m", "axis"])
def polyder(c: ArrayLike, m: int = 1, scl: float = 1, axis: int = 0) -> jax.Array:
    """
    Differentiate a polynomial (or N-D grid of polynomials).
    JAX analog to numpy.polynomial.polynomial.polyder.
    """
    c = jnp.atleast_1d(c)
    if m < 0:
        raise ValueError("m must be non-negative")
    if m == 0:
        return c

    n = c.shape[axis]
    if n <= m:
        # If differentiating more times than the degree, return a zero polynomial
        shape = list(c.shape)
        shape[axis] = 1
        return jnp.zeros(shape, dtype=c.dtype)

    indices = jnp.arange(m, n)
    c_der = jnp.take(c, indices, axis=axis)

    k = np.arange(m, n)
    weights = np.ones_like(k)
    for i in range(m):
        weights = weights * (k - i)

    # Reshape weights so they broadcast against the target axis
    shape = [1] * c.ndim
    shape[axis] = -1
    weights = jnp.array(weights, dtype=c.dtype).reshape(shape)

    c_der = c_der * weights

    return c_der * (scl**m)


@partial(jax.jit, static_argnames=["deg", "opt"])
def _multivariate_polyfit_jit(
    x: ArrayLike,
    vals: ArrayLike,
    deg: tuple[int, ...],
    length_scale: float,
    opt: PolynomialFitSolverFactory,
) -> jax.Array:
    x = jnp.asarray(x)
    vals = jnp.asarray(vals)

    if x.ndim == 1:
        x = x[:, None]

    dim = len(deg)
    shape = tuple(d + 1 for d in deg)

    x_scaled = x / length_scale

    N_samples = x.shape[0]

    a = (x_scaled[:, 0, None] ** jnp.arange(deg[0] + 1)).reshape(
        [N_samples, deg[0] + 1] + [1] * (dim - 1)
    )

    # Broadcast-multiply the remaining dimensions
    for i in range(1, dim):
        v_i = (x_scaled[:, i, None] ** jnp.arange(deg[i] + 1)).reshape(
            [N_samples] + [1] * i + [deg[i] + 1] + [1] * (dim - i - 1)
        )
        a = a * v_i

    a = a.reshape(N_samples, -1)
    b = vals

    coeffs_flat = opt(deg)(a, b)

    scale_powers = sum(
        jnp.arange(d + 1).reshape([1] * i + [d + 1] + [1] * (dim - i - 1))
        for i, d in enumerate(deg)
    )

    # Reshape back to ND array and apply length-scale division
    return coeffs_flat.reshape(shape) / (length_scale**scale_powers)


def multivariate_polyfit(
    x: ArrayLike,
    vals: ArrayLike,
    deg: Sequence[int],
    l: float = 1.0,  # noqa: E741
    opt: PolynomialFitSolverFactory = dflt_ls_opt,
) -> jax.Array:
    """
    Fits a set of data with a multivariate polynomial.

    Args:
        x (jax.Array | np.ndarray): Independent values of shape (N, dim).
        vals (jax.Array | np.ndarray): Dependent value of shape (N,).
        deg (Sequence[int]): Degree of polynomial used in the fit.
        l (float): Length scale used when fitting, defaults to 1.
        opt (Callable): Generator of the appropriate optimization function for the fit.

    Returns:
        (jax.Array): Coefficients of the best fit multivariate polynomial.
    """
    # By converting `deg` to a standard tuple, we ensure it is hashable
    deg_tuple = tuple(int(d) for d in deg)

    return _multivariate_polyfit_jit(x, vals, deg_tuple, l, opt)


@jax.jit
def polyval(x: ArrayLike, alpha: ArrayLike) -> jax.Array:
    x = jnp.asarray(x)
    alpha = jnp.asarray(alpha)
    dim = alpha.ndim

    if x.ndim == 1:
        x = x[:, None] if dim == 1 else x[None, :]

    powers = jnp.asarray(np.indices(alpha.shape).reshape(dim, -1).T)
    terms = jnp.prod(x[:, None, :] ** powers[None, :, :], axis=-1)
    return jnp.sum(terms * alpha.reshape(-1), axis=-1)
