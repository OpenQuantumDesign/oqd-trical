import numpy as np
import torch
from ..misc.linalg import gram_schimdt, random_unitary


def control_eigenfreqs(
    ti,
    target_w,
    guess_b=None,
    num_iter=1000,
    dir="x",
    term_tol=(1e-5, 0.0),
    det_tol=1e-2,
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[dir]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    w = ti.w_pa[a * N : (a + 1) * N]

    w_scale = np.linalg.norm(target_w)
    ndA = A / w_scale ** 2
    target_ndw = target_w / w_scale

    if guess_b is None:
        _b = 2 * np.random.rand(N, N) - 1
        _b = gram_schimdt(_b)
        while not np.isclose(np.matmul(_b, _b.transpose()), np.eye(N)).all():
            _b = 2 * np.random.rand(N, N) - 1
            _b = gram_schimdt(_b)
    else:
        _b = guess_b

    for i in range(num_iter):
        _ndA = np.einsum("im,m,mj->ij", _b, target_ndw ** 2, _b.transpose())

        if i == 0:
            ndAs = np.copy(_ndA.reshape(1, *_ndA.shape))
        else:
            if (_w >= 0).all() and np.isclose(
                np.sqrt(_w), target_w, rtol=term_tol[0], atol=term_tol[1]
            ).all():
                print("Terminated at iteration {}".format(i + 1))
                break
            ndAs = np.concatenate((ndAs, _ndA.reshape(1, *_ndA.shape)), axis=0)

        idcs = np.triu_indices(N, k=1)

        _ndAt = np.copy(_ndA)
        _ndAt[idcs] = _ndAt[idcs[1], idcs[0]] = np.copy(ndA[idcs])

        _ndw, _b = np.linalg.eigh(_ndAt)
        idcs = np.argsort(-_ndw)
        _ndw = _ndw[idcs]
        _w = _ndw * w_scale ** 2
        _b = _b[:, idcs]

        if np.abs(np.linalg.det(_b ** 2)) < det_tol:
            print("possible cycling at iteration {}".format(i))
            _ndAt[range(N), range(N)] = (
                np.random.rand(N) * _ndAt[range(N), range(N)].max()
            )

            _ndw, _b = np.linalg.eigh(_ndAt)
            idcs = np.argsort(-_ndw)
            _ndw = _ndw[idcs]
            _w = _ndw * w_scale ** 2
            _b = _b[:, idcs]

        if i == 0:
            ndAts = np.copy(_ndAt.reshape(1, *_ndAt.shape))
        else:
            ndAts = np.concatenate((ndAts, _ndAt.reshape(1, *_ndAt.shape)), axis=0)

    _A = _ndA * w_scale ** 2
    _At = _ndAt * w_scale ** 2
    As = ndAs * w_scale ** 2
    Ats = ndAts * w_scale ** 2
    omega_opt = np.sqrt(np.abs(np.diag(_At - A)))
    omega_opt_sign = np.sign(np.diag(_At - A))

    return (omega_opt, omega_opt_sign), np.sqrt(_w) - target_w, As, Ats


def control_eigenvecs(
    ti, target_b, num_iter=1000, dir="y", ttol=(1e-3, 0.0), ctol=(1e-10, 0.0)
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[dir]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    w = ti.w_pa[a * N : (a + 1) * N]

    last_initialization = 0
    _w = np.copy(w) ** 2

    for i in range(num_iter):
        _A = np.einsum("im,m,mj->ij", target_b, _w, target_b.transpose())

        if i == 0:
            As = np.copy(_A.reshape(1, *_A.shape))
        else:
            if np.isclose(
                np.abs(np.diag(np.matmul(_b.transpose(), target_b))),
                np.ones(N),
                rtol=ttol[0],
                atol=ttol[1],
            ).all():
                print(
                    "Terminated at iteration {} and iteration {} from reinitialization".format(
                        i + 1, i - last_initialization + 1
                    )
                )
                break

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

    omega_opt = np.sqrt(np.abs(np.diag(_At - A)))
    omega_opt_sign = np.sign(np.diag(_At - A))

    return (
        (omega_opt, omega_opt_sign),
        np.abs(np.matmul(_b.transpose(), target_b)),
        As,
        Ats,
    )


def multi_inst_control_eigenfreqs(
    ti, target_w, num_inst=1000, num_iter=1000, dir="x", det_tol=1e-2,
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[dir]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

    _b = random_unitary(num_inst, N)

    for i in range(num_iter):
        _A = np.einsum("...im,m,...jm->...ij", _b, target_w ** 2, _b)

        idcs = np.triu_indices(N, k=1)
        _At = np.copy(_A)
        _At[:, idcs[0], idcs[1]] = _At[:, idcs[1], idcs[0]] = np.copy(A[idcs])
        _w, _b = np.linalg.eigh(_At)
        _w = np.flip(_w, -1)
        _b = np.flip(_b, -1)
        dets = np.linalg.det(_b ** 2)
        _b = _b[np.abs(dets) > det_tol]

    omega_opt = np.sqrt(np.abs(_At[:, range(N), range(N)] - A[range(N), range(N)]))
    omega_opt_sign = np.sign(_At[:, range(N), range(N)] - A[range(N), range(N)])

    return (omega_opt, omega_opt_sign), np.sqrt(_w) - target_w


def generate_control_eigenfreqs_residue(target_w, ti, dir="x"):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[dir]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

    def residue(A_opt):
        _A = np.copy(A)
        _A[range(N), range(N)] = _A[range(N), range(N)] + A_opt
        _w = np.sqrt(np.linalg.eigh(_A)[0])
        _w = -np.sort(-_w)
        return np.linalg.norm(_w - target_w)

    return residue


def multi_control_eigenfreqs(
    ti,
    target_w,
    guess_b=None,
    num_iter=1000,
    dir="x",
    term_tol=(1e-5, 0.0),
    det_tol=1e-2,
):
    M = len(target_w)

    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[dir]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    w = ti.w_pa[a * N : (a + 1) * N]

    w_scale = np.linalg.norm(target_w, axis=-1)
    ndA = np.copy(A)
    ndA = np.tile(A, (M, 1, 1))
    ndA = A / (w_scale ** 2).reshape(-1, 1, 1)
    target_ndw = target_w / (w_scale).reshape(-1, 1)

    if guess_b is None:
        _b = random_unitary(M, N)
    else:
        _b = guess_b

    completed_idcs = np.zeros(M, dtype=bool)
    completed_iter = np.full(M, np.nan)
    for i in range(num_iter):
        print(i, completed_idcs.sum())
        if i == 0:
            _ndA = np.zeros((M, N, N))

        _ndA[np.logical_not(completed_idcs)] = np.einsum(
            "...im,...m,...mj->...ij",
            _b[np.logical_not(completed_idcs)],
            target_ndw[np.logical_not(completed_idcs)] ** 2,
            _b[np.logical_not(completed_idcs)].swapaxes(-1, -2),
        )
        idcs = np.triu_indices(N, k=1)

        _ndAt = np.copy(_ndA)
        _ndAt[:, idcs[0], idcs[1]] = _ndAt[:, idcs[1], idcs[0]] = np.copy(
            ndA[:, idcs[0], idcs[1]]
        )

        _ndw, _b = np.linalg.eigh(_ndAt)
        _ndw = np.flip(_ndw, axis=-1)
        _w = _ndw * (w_scale ** 2).reshape(-1, 1)
        _b = np.flip(_b, axis=-1)

        completed_idcs = np.logical_or(
            completed_idcs,
            np.isclose(np.sqrt(_w), target_w, rtol=term_tol[0], atol=term_tol[1]).all(
                axis=-1
            ),
        )
        completed_iter[
            np.logical_and(
                np.isnan(completed_iter),
                np.isclose(
                    np.sqrt(_w), target_w, rtol=term_tol[0], atol=term_tol[1]
                ).all(axis=-1),
            )
        ] = (i + 1)

        reinit_idcs = np.abs(np.linalg.det(_b ** 2)) < det_tol
        if reinit_idcs.sum() > 0:
            _ndAt[reinit_idcs] = np.random.uniform(
                0,
                np.tile(
                    _ndAt[reinit_idcs][:, range(N), range(N)].max(axis=-1), (N, N, 1)
                ).transpose(),
            )
            _ndAt[:, idcs[0], idcs[1]] = _ndAt[:, idcs[1], idcs[0]] = np.copy(
                ndA[:, idcs[0], idcs[1]]
            )

            _ndw, _b = np.linalg.eigh(_ndAt)
            _ndw = np.flip(_ndw, axis=-1)
            _w = _ndw * (w_scale ** 2).reshape(-1, 1)
            _b = np.flip(_b, axis=-1)

    _A = _ndA * (w_scale ** 2).reshape(-1, 1, 1)
    _At = _ndAt * (w_scale ** 2).reshape(-1, 1, 1)
    omega_opt = np.sqrt(np.abs(_At[:, range(N), range(N)] - A[range(N), range(N)]))
    omega_opt_sign = np.sign(_At[:, range(N), range(N)] - A[range(N), range(N)])

    return (omega_opt, omega_opt_sign), np.sqrt(_w) - target_w, completed_iter
