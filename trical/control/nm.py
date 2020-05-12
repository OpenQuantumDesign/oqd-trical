import numpy as np
import torch
from ..misc.linalg import gram_schimdt, random_unitary


def control_eigenfreqs(
    ti,
    target_w,
    guess_b=None,
    num_iters=1000,
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

    w_scale = w.max()
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

    for i in range(num_iters):
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
            r = 2 * np.random.rand(N) - 1
            r = r / np.linalg.norm(r)
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
    ti, target_b, num_iters=1000, dir="y", ttol=(1e-3, 0.0), ctol=(1e-10, 0.0)
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[dir]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    w = ti.w_pa[a * N : (a + 1) * N]

    last_initialization = 0
    _w = np.copy(w) ** 2

    for i in range(num_iters):
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

            # rep = (
            #     np.isclose(As, _A, rtol=ctol[0], atol=ctol[1]).all(axis=-1).all(axis=-1)
            # )
            # if rep.any():
            #     period = len(As) - np.array(np.where(rep == 1)).max()
            #     print(
            #         "Cycling with period {} detected at iteration {}".format(period, i)
            #     )
            #     last_initialization = i

            #     _w = np.random.normal(1, 0.2) * w
            #     _w = _w[np.argsort(-_w)]

            #     _A = np.einsum("im,m,mj->ij", target_b, _w, target_b.transpose())

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


def mi_control_eigenfreqs(
    ti, target_w, num_inst=1000, num_iters=1000, dir="x",
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[dir]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

    _b = random_unitary(num_inst, N)

    for i in range(num_iters):
        _A = np.einsum("...im,m,...jm->...ij", _b, target_w ** 2, _b)

        if i == 0:
            As = np.copy(_A.reshape(1, *_A.shape))
        else:
            As = np.concatenate((As, _A.reshape(1, *_A.shape)), axis=0)

        idcs = np.triu_indices(N, k=1)

        _At = np.copy(_A)
        _At[:, idcs[0], idcs[1]] = _At[:, idcs[1], idcs[0]] = np.copy(A[idcs])
        _w, _b = np.linalg.eigh(_At)
        _w = np.flip(_w, -1)
        _b = np.flip(_b, -1)

        if i == 0:
            Ats = np.copy(_At.reshape(1, *_At.shape))
        else:
            Ats = np.concatenate((Ats, _At.reshape(1, *_At.shape)), axis=0)

    omega_opt = np.sqrt(np.abs(_At[:, range(N), range(N)] - A[range(N), range(N)]))
    omega_opt_sign = np.sign(_At[:, range(N), range(N)] - A[range(N), range(N)])

    return (omega_opt, omega_opt_sign), np.sqrt(_w) - target_w, As, Ats


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def multi_control_eigenfreqs(
    ti, target_w, num_iters=1000, dir="x",
):
    cuda = torch.device("cuda")
    torch.cuda.init()

    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[dir]
    N = ti.N
    A = torch.from_numpy(ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]).to(device=cuda)

    w = torch.from_numpy(ti.w_pa[a * N : (a + 1) * N]).to(device=cuda) ** 2
    _b = torch.from_numpy(
        (
            ti.b_pa[a * N : (a + 1) * N, a * N : (a + 1) * N]
            if type(ti.m) == float
            else np.einsum(
                "im,i->im", ti.b_pa[a * N : (a + 1) * N, a * N : (a + 1) * N], ti.m
            )
        ),
    ).to(device=cuda)

    target_w = torch.from_numpy(target_w).to(device=cuda)

    sigma = sigmoid(np.linspace(10, -10, num_iters))

    for i in range(num_iters):
        print(i)
        _w = target_w ** 2
        try:
            _A = torch.einsum("...im,...m,...mj->...ij", (_b, _w, _b.transpose(-1, -2)))
        except:
            _A = torch.einsum("im,...m,mj->...ij", (_b, _w, _b.transpose(0, 1)))

        if i == 0:
            As = _A.view(1, *_A.shape).clone()
        else:
            As = torch.cat((As, _A.view(1, *_A.shape)), axis=0)

        idcs = np.triu_indices(N, k=1)

        _At = _A.clone()
        _At[:, idcs[0], idcs[1]] = _At[:, idcs[1], idcs[0]] = torch.normal(
            A[idcs], A[idcs] * sigma[i]
        )

        if i == 0:
            Ats = _At.view(1, *_At.shape).clone()
        else:
            Ats = torch.cat((Ats, _At.view(1, *_At.shape)), axis=0)

        _e = torch.symeig(_At, eigenvectors=True)
        _w = _e.eigenvalues
        _b = _e.eigenvectors
        _w = torch.flip(_w, dims=(-1,))
        _b = torch.flip(_b, dims=(-1,))

    omega_opt = torch.sqrt(torch.abs((_At - A)[:, range(N), range(N)]))
    omega_opt_sign = torch.sign((_At - A)[:, range(N), range(N)])

    return (omega_opt, omega_opt_sign), torch.sqrt(_w) - target_w, As, Ats


def generate_control_eigenfreqs_residue(target_w, ti, dir="x"):
    N = len(target_w)
    a = {"x": 0, "y": 1, "z": 2}[dir]
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

    def residue(A_opt):
        _A = np.copy(A)
        _A[range(N), range(N)] = _A[range(N), range(N)] + A_opt
        _w = np.sqrt(np.linalg.eigh(_A)[0])
        _w = -np.sort(-_w)
        return np.linalg.norm(_w - target_w)

    return residue
