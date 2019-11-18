import numpy as np
from scipy import optimize as opt


def dflt_opt(N, dim):
    if dim == 1:
        x_guess = np.linspace(-(N - 1) / 2, (N - 1) / 2, N)
    else:
        x_guess = np.append(
            np.concatenate([np.zeros(N)] * (dim - 1)),
            np.linspace(-(N - 1) / 2, (N - 1) / 2, N),
        )
    return lambda f: opt.minimize(f, x_guess, method="SLSQP").x
