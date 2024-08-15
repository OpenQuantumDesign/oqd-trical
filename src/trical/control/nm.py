import itertools as itr
import numpy as np
from ..misc.linalg import gram_schimdt, random_unitary


def control_eigenfreqs(
    ti,
    target_w,
    guess_b=None,
    maxiter=1000,
    direc="x",
    term_tol=(0.0, 10.0),
    det_tol=1e-2,
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

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
        _b = np.copy(guess_b)

    for i in range(maxiter):
        _ndA = np.einsum("im,m,mj->ij", _b, target_ndw ** 2, _b.transpose())

        if i == 0:
            ndAs = np.copy(_ndA.reshape(1, *_ndA.shape))
        else:
            if (_w >= 0).all() and np.isclose(
                np.sqrt(_w), target_w, rtol=term_tol[0], atol=term_tol[1]
            ).all():
                print("Terminated at iteration {}".format(i))
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
            r = np.random.rand(N)
            r = r / r.sum()
            _ndAt[range(N), range(N)] = r

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
    ti, target_b, maxiter=1000, direc="y", ttol=(1e-3, 0.0), ctol=(1e-10, 0.0)
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    w = ti.w_pa[a * N : (a + 1) * N]

    last_initialization = 0
    _w = np.copy(w) ** 2

    for i in range(maxiter):
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
                print("Terminated at iteration {}".format(i))
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


def generate_control_eigenfreqs_residue(target_w, ti, direc="x"):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
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
    maxiter=1000,
    direc="x",
    term_tol=(0.0, 10.0),
    det_tol=1e-2,
    show_progress=True,
):
    M = len(target_w)

    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

    w_scale = np.linalg.norm(target_w, axis=-1)
    ndA = np.copy(A)
    ndA = np.tile(A, (M, 1, 1))
    ndA = A / (w_scale ** 2).reshape(-1, 1, 1)
    target_ndw = target_w / (w_scale).reshape(-1, 1)

    if guess_b is None:
        _b = random_unitary(M, N)
    else:
        _b = np.copy(guess_b)

    completed_idcs = np.zeros(M, dtype=bool)
    completed_iter = np.full(M, np.nan)
    for i in range(maxiter):
        if show_progress:
            print("{:<10}{:<10}".format(i, completed_idcs.sum()))
        if i == 0:
            _ndA = np.einsum(
                "...im,...m,...mj->...ij", _b, target_ndw ** 2, _b.swapaxes(-1, -2)
            )

        else:
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
        ] = i

        reinit_idcs = np.abs(np.linalg.det(_b ** 2)) < det_tol
        if reinit_idcs.sum() > 0:
            r = np.random.rand(reinit_idcs.sum(), N, N)
            r = r / np.trace(r, axis1=-1, axis2=-2).reshape(-1, 1, 1)
            _ndAt[reinit_idcs] = r
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

    return (omega_opt, omega_opt_sign), np.sqrt(_w) - target_w, _A, _At, completed_iter


def control_eigenfreqs_step(
    ti,
    target_w,
    initial_A=None,
    guess_b=None,
    maxiter=1000,
    direc="x",
    term_tol=(0.0, 10.0),
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N

    if initial_A is None:
        A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    else:
        A = initial_A

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
        _b = np.copy(guess_b)

    for i in range(maxiter):
        _ndA = np.einsum("im,m,mj->ij", _b, target_ndw ** 2, _b.transpose())

        if i == 0:
            ndAs = np.copy(_ndA.reshape(1, *_ndA.shape))
        else:
            if (_w >= 0).all() and np.isclose(
                np.sqrt(_w), target_w, rtol=term_tol[0], atol=term_tol[1]
            ).all():
                print("Terminated at iteration {}".format(i))
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

        if i == 0:
            ndAts = np.copy(_ndAt.reshape(1, *_ndAt.shape))
        else:
            ndAts = np.concatenate((ndAts, _ndAt.reshape(1, *_ndAt.shape)), axis=0)

    _A = _ndA * w_scale ** 2
    _At = _ndAt * w_scale ** 2
    As = ndAs * w_scale ** 2
    Ats = ndAts * w_scale ** 2

    return np.sqrt(_w) - target_w, As, Ats


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


dflt_epsilon = sigmoid(np.linspace(-5, 5, 1001))
dflt_epsilon = (
    (dflt_epsilon - dflt_epsilon.min()) / (dflt_epsilon.max() - dflt_epsilon.min())
)[1:]


def control_eigenfreqs_gradual(
    ti, target_w, epsilon=dflt_epsilon, maxiter=1000, direc="x", term_tol=(0.0, 10.0)
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    w = ti.w_pa[a * N : (a + 1) * N]
    b = (
        ti.b_pa[a * N : (a + 1) * N, a * N : (a + 1) * N]
        if type(ti.m) == float
        else ti.b_pa[a * N : (a + 1) * N, a * N : (a + 1) * N]
        * np.sqrt(ti.m).reshape(-1, 1)
    )

    for i in range(len(epsilon)):
        if i == 0:
            delta_w, As, Ats = control_eigenfreqs_step(
                ti,
                epsilon[i] * target_w + (1 - epsilon[i]) * w,
                guess_b=b,
                maxiter=maxiter,
                direc=direc,
                term_tol=term_tol,
            )
        else:
            delta_w, _As, _Ats = control_eigenfreqs_step(
                ti,
                epsilon[i] * target_w + (1 - epsilon[i]) * np.sqrt(_w),
                initial_A=Ats[-1],
                guess_b=_b,
                maxiter=maxiter,
                direc=direc,
                term_tol=term_tol,
            )
            As = np.concatenate((As, _As), axis=0)
            Ats = np.concatenate((Ats, _Ats), axis=0)

        _w, _b = np.linalg.eigh(Ats[-1])
        _w = np.flip(_w)
        _b = np.flip(_b, axis=-1)

        print(
            "{:<10}{:<20}{:<20}".format(
                i,
                "{:.5e}".format(np.linalg.norm(np.sqrt(_w) - target_w)),
                "{:.5e}".format(np.linalg.norm(delta_w)),
            )
        )

    omega_opt = np.sqrt(np.abs(np.diag(Ats[-1] - A)))
    omega_opt_sign = np.sign(np.diag(Ats[-1] - A))

    return (omega_opt, omega_opt_sign), np.sqrt(_w) - target_w, As, Ats


def multi_inst_control_eigenfreqs(
    ti, target_w, guess_b=None, num_inst=1000, maxiter=1000, direc="x"
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

    if guess_b is None:
        _b = random_unitary(num_inst, N)
    else:
        _b = guess_b

    for i in range(maxiter):
        _A = np.einsum("...im,m,...jm->...ij", _b, target_w ** 2, _b)

        idcs = np.triu_indices(N, k=1)
        _At = np.copy(_A)
        _At[:, idcs[0], idcs[1]] = _At[:, idcs[1], idcs[0]] = np.copy(A[idcs])
        _w, _b = np.linalg.eigh(_At)
        _w = np.flip(_w, -1)
        _b = np.flip(_b, -1)

    A_diag = _At[:, range(N), range(N)] - A[range(N), range(N)]

    return _At, np.sqrt(_w, dtype=np.complex) - target_w


def control_eigenfreqs_pop(
    ti,
    target_w,
    mutation=0.1,
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
        At, delta_w = multi_inst_control_eigenfreqs(
            ti, target_w, _b, popsize, maxiter, direc
        )

        if i == 0:
            Ats = np.copy(At).reshape(1, *At.shape)
        else:
            Ats = np.concatenate((Ats, np.copy(At).reshape(1, *At.shape)), axis=0)

        ndelta_w = np.linalg.norm(delta_w, axis=-1)
        print("{:<10}{:<20}".format(i, "{:.5e}".format(ndelta_w.min())))
        minidx = ndelta_w.argmin()

        if np.isclose(
            target_w + delta_w[minidx], target_w, rtol=term_tol[0], atol=term_tol[1]
        ).all():
            break

        if i < popsteps:
            e = list(range(popsize))
            e.pop(minidx)

            cidcs = np.fromiter(itr.chain(*itr.combinations(e, 2)), dtype=int).reshape(
                -1, 2
            )

            ridcs = np.arange(len(cidcs), dtype=int)
            np.random.shuffle(ridcs)
            ridcs = ridcs[:popsize]

            A_diag = At[:, range(N), range(N)]
            A_diag = A_diag[minidx] + mutation * (
                A_diag[cidcs[ridcs, 0]] - A_diag[cidcs[ridcs, 1]]
            )
            At[:, range(N), range(N)] = A_diag

            _w, _b = np.linalg.eigh(At)
            _w = np.flip(_w, axis=-1)
            _b = np.flip(_b, axis=-1)

    omega_opt = np.sqrt(np.abs(np.diag(At[minidx] - A)))
    omega_opt_sign = np.sign(np.diag(At[minidx] - A))

    return (omega_opt, omega_opt_sign), delta_w[minidx], Ats


def control_eigenfreqs_de(
    ti,
    target_w,
    mutation=0.1,
    greed=0.1,
    crossover=0.1,
    maxiter=1000,
    popsteps=100,
    popsize=50,
    direc="x",
    term_tol=(0.0, 2e1 * np.pi),
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]

    _b = random_unitary(popsize, N)

    for i in range(popsteps):
        if i == 0:
            At, delta_w = multi_inst_control_eigenfreqs(
                ti, target_w, _b, popsize, maxiter, direc
            )
            Ats = np.copy(At).reshape(1, *At.shape)

            ndelta_w = np.linalg.norm(delta_w, axis=-1)
            minidx = ndelta_w.argmin()

        idcs = np.tile(np.arange(popsize - 1), (popsize, 1))
        idcs[np.triu_indices(popsize - 1, k=0)] += 1
        [np.random.shuffle(j) for j in idcs]

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

        At2, delta_w2 = multi_inst_control_eigenfreqs(
            ti, target_w, _b, popsize, maxiter, direc
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


def sort_btb(btb):
    idcs = []
    for i in range(len(btb)):
        rng = range(len(btb))
        rng = np.delete(rng, idcs)
        _btb = btb[:, rng]
        idcs.append(rng[_btb[i].argmax()])
    idcs = np.array(idcs)
    return idcs


def multi_inst_control_eigenvecs(
    ti, target_b, guess_w=None, scale_w=1.0, num_inst=1000, maxiter=1000, direc="x"
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    w = ti.w_pa[a * N : (a + 1) * N]

    if guess_w is None:
        _w = np.random.rand(num_inst, N)
        _w = _w / np.linalg.norm(_w, axis=-1).reshape(-1, 1)
        _w = (_w * np.linalg.norm(w) * scale_w) ** 2
    else:
        _w = np.copy(guess_w)

    for i in range(maxiter):
        _A = np.einsum("im,...m,jm->...ij", target_b, _w, target_b)

        idcs = np.triu_indices(N, k=1)
        _At = np.copy(_A)
        _At[:, idcs[0], idcs[1]] = _At[:, idcs[1], idcs[0]] = np.copy(A[idcs])
        _w, _b = np.linalg.eigh(_At)

        btb = np.abs(np.einsum("mi,...in->...mn", target_b.transpose(), _b))
        idcs = np.argmax(btb, axis=-1)

        for j in range(num_inst):
            if len(np.unique(idcs[j])) != N:
                idcs[j] = sort_btb(btb[j])
            _w[j] = _w[j, idcs[j]]
            _b[j] = _b[j, :, idcs[j]].transpose()

    A_diag = _At[:, range(N), range(N)] - A[range(N), range(N)]

    return (
        _At,
        np.ones(N)
        - np.abs(np.einsum("mi,...in->...mn", target_b.transpose(), _b))[
            :, range(N), range(N)
        ],
    )


def control_eigenvecs_de(
    ti,
    target_b,
    scale_w=1.0,
    mutation=0.1,
    greed=0.1,
    crossover=0.1,
    maxiter=1000,
    popsteps=100,
    popsize=50,
    direc="x",
    term_tol=(0.0, 1e-5),
):
    if np.isin(np.array(["w_pa", "b_pa"]), np.array(ti.__dict__.keys())).sum() != 2:
        ti.principle_axis()

    a = {"x": 0, "y": 1, "z": 2}[direc]
    N = ti.N
    A = ti.A[a * N : (a + 1) * N, a * N : (a + 1) * N]
    w = ti.w[a * N : (a + 1) * N]

    _w = np.random.rand(popsize, N)
    _w = _w / np.linalg.norm(_w, axis=-1).reshape(-1, 1)
    _w = (_w * np.linalg.norm(w) * scale_w) ** 2

    for i in range(popsteps):
        if i == 0:
            At, delta_b = multi_inst_control_eigenvecs(
                ti, target_b, _w, scale_w, popsize, maxiter, direc
            )
            Ats = np.copy(At).reshape(1, *At.shape)

            ndelta_b = np.linalg.norm(delta_b, axis=-1)
            minidx = ndelta_b.argmin()

        idcs = np.tile(np.arange(popsize - 1), (popsize, 1))
        idcs[np.triu_indices(popsize - 1, k=0)] += 1
        [np.random.shuffle(j) for j in idcs]

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

        btb = np.abs(np.einsum("mi,...in->...mn", target_b.transpose(), _b))
        idcs = np.argmax(btb, axis=-1)

        for j in range(popsize):
            if len(np.unique(idcs[j])) != N:
                idcs[j] = sort_btb(btb[j])
            _w[j] = _w[j, idcs[j]]
            _b[j] = _b[j][:, idcs[j]]

        At2, delta_b2 = multi_inst_control_eigenvecs(
            ti, target_b, _w, scale_w, popsize, maxiter, direc
        )

        p2 = (
            np.linalg.norm(delta_b2, axis=-1) < np.linalg.norm(delta_b, axis=-1)
        ).reshape(-1, 1, 1)
        At = np.logical_not(p2) * At + p2 * At2

        p2 = p2.reshape(-1, 1)
        delta_b = np.logical_not(p2) * delta_b + p2 * delta_b2

        ndelta_b = np.linalg.norm(delta_b, axis=-1)
        print("{:<10}{:<20}".format(i, "{:.5e}".format(ndelta_b.min())))
        minidx = ndelta_b.argmin()

        Ats = np.concatenate((Ats, np.copy(At).reshape(1, *At.shape)), axis=0)

        if np.isclose(
            delta_b[minidx], np.zeros(N), rtol=term_tol[0], atol=term_tol[1]
        ).all():
            break

    omega_opt = np.sqrt(np.abs(np.diag(At[minidx] - A)))
    omega_opt_sign = np.sign(np.diag(At[minidx] - A))

    return (omega_opt, omega_opt_sign), delta_b[minidx], Ats
