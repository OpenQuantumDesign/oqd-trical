import numpy as np


def sort_btb(btb):
    idcs = []
    for i in range(len(btb)):
        rng = range(len(btb))
        rng = np.delete(rng, idcs)
        _btb = btb[:, rng]
        idcs.append(rng[_btb[i].argmax()])
    idcs = np.array(idcs)
    return idcs


def mi(
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

        for i in range(num_inst):
            if len(np.unique(idcs[i])) != N:
                idcs[i] = sort_b(btb[i])
            _w[i] = _w[i, idcs[i]]
            _b[i] = _b[i, :, idcs[i]].transpose()

    A_diag = _At[:, range(N), range(N)] - A[range(N), range(N)]

    return (
        _At,
        np.ones(N)
        - np.abs(np.einsum("mi,...in->...mn", target_b.transpose(), _b))[
            :, range(N), range(N)
        ],
    )


def de(
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
            At, delta_b = mi(
                ti, target_b, _w, scale_w, popsize, maxiter, direc
            )
            Ats = np.copy(At).reshape(1, *At.shape)

            ndelta_b = np.linalg.norm(delta_b, axis=-1)
            minidx = ndelta_b.argmin()

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

        btb = np.abs(np.einsum("mi,...in->...mn", target_b.transpose(), _b))
        idcs = np.argmax(btb, axis=-1)

        for j in range(popsize):
            if len(np.unique(idcs[j])) != N:
                idcs[j] = sort_b(btb[j])
            _w[j] = _w[j, idcs[j]]
            _b[j] = _b[j][:, idcs[j]]

        At2, delta_b2 = mi(
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
