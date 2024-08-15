import numpy as np
from ..misc.linalg import random_unitary


def mi(ti, target_w, **kwargs):
    params = {
        "direc": "x",
        "guess_b": None,
        "maxiter": 1000,
        "popsize": 1000,
        "torch": False,
        "torch_params": {"dtype": torch.float, "device": torch.device("cpu")},
    }
    params.update(kwargs)

    if (
        np.isin(np.array(["b_pa", "w_pa", "x_pa"]), np.array(ti.__dict__.keys())).sum()
        != 3
    ):
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[params["direc"]]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

    if params["guess_b"] is None:
        _b = random_unitary(params["popsize"], N)
    else:
        _b = params["guess_b"]

    for i in range(maxiter):
        _A = np.einsum("...im,m,...jm->...ij", _b, target_w ** 2, _b)

        idcs = np.triu_indices(N, k=1)
        _At = np.copy(_A)
        _At[:, idcs[0], idcs[1]] = _At[:, idcs[1], idcs[0]] = np.copy(A[idcs])
        _w, _b = np.linalg.eigh(_At)
        _w = np.flip(_w, -1)
        _b = np.flip(_b, -1)

    A_diag = _At[:, range(N), range(N)] - A[range(N), range(N)]

    return _At, np.sqrt(_w) - target_w


def de(
    ti,
    target_w,
    mutation=0.1,
    greed=0.1,
    crossover=0.1,
    maxiter=1000,
    popsteps=100,
    popsize=50,
    direc="x",
    term_tol=(0.0, 10.0),
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

    _b = random_unitary(popsize, N)

    for i in range(popsteps):
        if i == 0:
            At, delta_w = mi(ti, target_w, guess_b=_b, popsize=popsize, maxiter=maxiter, direc=direc)
            Ats = np.copy(At).reshape(1, *At.shape)

            ndelta_w = np.linalg.norm(delta_w, axis=-1)
            minidx = ndelta_w.argmin()

        idcs = np.tile(np.arange(popsize - 1), (popsize, 1))
        idcs[np.triu_indices(popsize - 1, k=0)] += 1
        [np.random.shuffle(i) for i in idcs]

        x = At[:, range(N), range(N)]

        v = (
            x[range(popsize)]
            + greed * (x[minidx] - x[range(popsize)])
            + mutation * (x[idcs[:, 0]] - x[idcs[:, 1]])
        )

        r = np.random.rand(*v.shape)
        p = r <= crossover

        u = np.logical_not(p) * x + p * v

        At2 = np.copy(At)
        At2[:, range(N), range(N)] = u

        _w, _b = np.linalg.eigh(At2)
        _w = np.flip(_w, axis=-1)
        _b = np.flip(_b, axis=-1)

        At2, delta_w2 = mi(
            ti, target_w, guess_b=_b, popsize=popsize, maxiter=maxiter, direc=direc
        )

        p2 = (
            np.linalg.norm(delta_w2, axis=-1) < np.linalg.norm(delta_w, axis=-1)
        ).reshape(-1, 1, 1)
        At = np.logical_not(p2) * At + p2 * At2

        p2 = p2.reshape(-1, 1)
        delta_w = np.logical_not(p2) * delta_w + p2 * delta_w2

        ndelta_w = np.linalg.norm(delta_w, axis=-1)
        print("{:<10}{:<20}".format(i, "{:.5e}".format(ndelta_w.min())))
        minidx = ndelta_w.argmin()

        Ats = np.concatenate((Ats, np.copy(At).reshape(1, *At.shape)), axis=0)

        if np.isclose(
            target_w + delta_w[minidx], target_w, rtol=term_tol[0], atol=term_tol[1]
        ).all():
            break

    omega_opt = np.sqrt(np.abs(np.diag(At[minidx] - A)))
    omega_opt_sign = np.sign(np.diag(At[minidx] - A))

    return (omega_opt, omega_opt_sign), delta_w[minidx], Ats
