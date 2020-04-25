import numpy as np
from ..misc.linalg import gram_schimdt


def control_eigenfreqs(
    ti, target_w, num_iters=1000, dir="x", ttol=(1e-5, 0.0), ctol=(1e-10, 0.0)
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[dir]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    b = (
        ti.b_pa[a * N : (a + 1) * N, a * N : (a + 1) * N]
        if type(ti.m) == float
        else np.einsum(
            "im,i->im", ti.b_pa[a * N : (a + 1) * N, a * N : (a + 1) * N], ti.m
        )
    )

    last_initialization = 0
    _b = np.copy(b)

    for i in range(num_iters):
        _A = np.einsum("im,m,mj->ij", _b, target_w ** 2, _b.transpose())

        if i == 0:
            As = np.copy(_A.reshape(1, *_A.shape))
        else:
            if (_w >= 0).all() and np.isclose(
                np.sqrt(_w), target_w, rtol=ttol[0], atol=ttol[1]
            ).all():
                print(
                    "Terminated at iteration {} and iteration {} from reinitialization".format(
                        i + 1, i - last_initialization + 1
                    )
                )
                break

            rep = (
                np.isclose(As, _A, rtol=ctol[0], atol=ctol[1]).all(axis=-1).all(axis=-1)
            )
            if rep.any():
                period = len(As) - np.array(np.where(rep == 1)).max()
                print(
                    "Cycling with period {} detected at iteration {}".format(period, i)
                )
                last_initialization = i

                _b = 2 * np.random.rand(N, N) - 1
                _b = gram_schimdt(_b)
                while not np.isclose(np.matmul(_b, _b.transpose()), np.eye(N)).all():
                    _b = 2 * np.random.rand(N, N) - 1
                    _b = gram_schimdt(_b)

                _A = np.einsum("im,m,mj->ij", _b, target_w ** 2, _b.transpose())

            As = np.concatenate((As, _A.reshape(1, *_A.shape)), axis=0)

        idcs = np.triu_indices(N, k=1)

        _At = np.copy(_A)
        _At[idcs] = np.copy(A[idcs])
        _At.transpose()[idcs] = np.copy(A.transpose()[idcs])

        if i == 0:
            Ats = np.copy(_At.reshape(1, *_At.shape))
        else:
            Ats = np.concatenate((Ats, _At.reshape(1, *_At.shape)), axis=0)

        _w, _b = np.linalg.eigh(_At)
        idcs = np.argsort(-_w)
        _w = _w[idcs]
        _b = _b[:, idcs]

        if i == num_iters - 1:
            print("Did not terminate")

    omega_opt = np.sqrt(np.abs(np.diag(_At - A)))
    omega_opt_sign = np.sign(np.diag(_At - A))

    return (omega_opt, omega_opt_sign), np.sqrt(_w) - target_w, As, Ats
