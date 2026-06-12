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
Module containing default optimization function generators for TrICal.
"""

########################################################################################
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import optax
from jax.typing import ArrayLike

########################################################################################


def dflt_opt(ti, **kwargs) -> Callable[[Callable[[jax.Array], jax.Array]], jax.Array]:
    """
    Default optimization function generator for equilibrium_position method of TrappedIons class.

    Args:
        ti (TrappedIons): Trapped ion system of interest.

    Returns:
        (Callable): Default optimization function that finds the equilibrium position of the trapped ions system of interest via the minimization of the potential.
    """
    # L-BFGS is the closest unconstrained quasi-Newton analogue to SLSQP natively in Optax
    opt_params = {"maxiter": 10000, "tol": 1e-15, "learning_rate": 0.1}
    opt_params.update(kwargs)

    if ti.dim == 1:
        x_guess = jnp.linspace(-(ti.N - 1) / 2, (ti.N - 1) / 2, ti.N)
    else:
        x_guess = jnp.append(
            jnp.concatenate([jnp.zeros(ti.N)] * (ti.dim - 1)),
            jnp.linspace(-(ti.N - 1) / 2, (ti.N - 1) / 2, ti.N),
        )

    def _dflt_opt(f):
        maxiter = opt_params["maxiter"]
        tol = opt_params["tol"]

        optimizer = optax.lbfgs(learning_rate=opt_params["learning_rate"])

        @jax.jit
        def step(params, opt_state):
            value, grad = jax.value_and_grad(f)(params)

            updates, opt_state = optimizer.update(
                grad, opt_state, params, value=value, grad=grad, value_fn=f
            )

            params = optax.apply_updates(params, updates)

            return params, opt_state, value, grad

        params = x_guess
        opt_state = optimizer.init(params)
        prev_value = jnp.inf

        for i in range(maxiter):
            params, opt_state, value, grad = step(params, opt_state)

            if jnp.abs(prev_value - value) < tol or jnp.max(jnp.abs(grad)) < tol:
                break

            prev_value = value

        assert not jnp.isnan(value), "Optimization diverged (NaNs encountered)."

        return params

    return _dflt_opt


def dflt_ls_opt(deg: Sequence[int]) -> Callable[[ArrayLike, ArrayLike], jax.Array]:
    """
    Default optimization function generator for multivariate_polyfit function.

    Args:
        deg (Sequence[int]): Degree of polynomial used in the fit.

    Returns:
        (Callable): Default optimization function that finds the best polynomial, of the specified degree, fit for the data .
    """

    @jax.jit
    def _dflt_ls_opt(a, b):
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        x, _, _, _ = jnp.linalg.lstsq(a, b, rcond=jnp.finfo(a.dtype).eps)
        return x

    return _dflt_ls_opt
