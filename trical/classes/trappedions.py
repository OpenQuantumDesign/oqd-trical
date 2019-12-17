from ..misc import constants as cst
from ..misc.linalg import norm, orthonormal_subset
from ..misc.optimize import dflt_opt
from .potential import CoulombPotential
from matplotlib import pyplot as plt
import numpy as np


class TrappedIons(object):
    def __init__(self, N, *ps, **kwargs):
        super(TrappedIons, self).__init__()

        params = {
            "dim": ps[0].dim if "dim" in ps[0].__dict__.keys() else 3,
            "l": 1e-6,
            "m": cst.m_a["Yb171"],
            "q": cst.e,
        }
        params.update(kwargs)
        self.__dict__.update(params)

        self.N = N
        self.ps = np.array(ps)

        self.cp = CoulombPotential(N, dim=self.dim, q=self.q)
        self.fp = self.cp + self.ps.sum()
        pass

    def equilibrium_position(self, opt=dflt_opt):
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        _ndfp = lambda x: ndfp(x.reshape(self.dim, self.N).transpose())

        self.x_ep = opt(self)(_ndfp).reshape(self.dim, self.N).transpose() * self.l
        return self.x_ep

    def normal_modes(self):
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        hess_phi = ndfp.hessian()

        if "x_ep" not in self.__dict__.keys():
            self.equilibrium_position()

        hess_phi_x_ep = hess_phi(self.x_ep / self.l)

        w, b = np.linalg.eigh(hess_phi_x_ep)
        w = np.sqrt(w * cst.k * cst.e ** 2 / (self.m * self.l ** 3))

        _b = np.round(np.copy(b), 3).transpose()
        s = (np.sign(_b).sum(1) >= 0) * 2 - 1
        b = np.einsum("n,in->in", s, b)

        idcs = np.argsort(np.flip(w, axis=0))

        self.w, self.b = w[idcs], b[:, idcs]
        return self.w, self.b

    def principle_axis(self, tol=1e-3):
        if np.isin(np.array(["w", "b"]), np.array(self.__dict__.keys())).sum() != 2:
            self.normal_modes()

        x_pa = orthonormal_subset(self.b.reshape(self.dim, -1).transpose(), tol=tol)

        assert len(x_pa) == self.dim

        _x_pa = np.round(np.copy(x_pa), 3)
        s = (np.sign(_x_pa).sum(1) >= 0) * 2 - 1
        x_pa = np.einsum("n,ni->ni", s, x_pa)

        _x_pa = np.round(np.copy(x_pa), 3)
        idcs = np.lexsort(
            np.concatenate((_x_pa.transpose(), np.abs(_x_pa).transpose()))
        )

        x_pa = x_pa[idcs]

        b_pa = np.einsum("ij,jn->in", x_pa, self.b.reshape(self.dim, -1)).reshape(
            self.dim * self.N, -1
        )
        w_pa = self.w

        _b = np.round(np.copy(b_pa), 3).transpose()
        s = (np.sign(_b).sum(1) >= 0) * 2 - 1
        b_pa = np.einsum("n,in->in", s, b_pa)

        n = np.round(
            np.array(
                [
                    norm(b_pa[i * self.N : (i + 1) * self.N].transpose())
                    for i in range(self.dim)
                ]
            )
        )
        idcs = np.lexsort(np.concatenate((np.flip(w_pa, axis=0).reshape(1, -1), n)))

        self.x_pa, self.w_pa, self.b_pa = x_pa, w_pa[idcs], b_pa[:, idcs]
        return self.x_pa, self.w_pa, self.b_pa

    def update_params(self, **kwargs):
        self.params.update(kwargs)
        self.__dict__.update(self.params)
        pass

    def plot_equilibrium_position(self, **kwargs):
        pass

    def plot_normal_modes(self, **kwargs):
        pass

    def plot_principle_axis(self, **kwargs):
        pass

    pass
