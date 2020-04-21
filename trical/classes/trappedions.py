from .base import Base
from ..misc import constants as cst
from ..misc.linalg import norm, orthonormal_subset
from ..misc.optimize import dflt_opt
from .potential import CoulombPotential
from matplotlib import pyplot as plt
import numpy as np


class TrappedIons(Base):
    """
    Object representing a system of trapped ions.

    :param N: Number of ions.
    :type N: :obj:`int`
    """

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
        """
        Function that calculates the equilibrium position of the ions.

        :param opt: Generator of the appropriate optimization function for finding the equilibrium position, defaults to :obj:`trical.misc.optimize.dflt_opt`
        :type opt: :obj:`types.FunctionType`, optional
        :returns: Equilibrium position of the ions.
        :rtype: :obj:`numpy.ndarray`
        """
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        _ndfp = lambda x: ndfp(x.reshape(self.dim, self.N).transpose())

        self.x_ep = opt(self)(_ndfp).reshape(self.dim, self.N).transpose() * self.l
        return self.x_ep

    def normal_modes(self):
        """
        Function that calculates the normal modes of the system.

        :Returns:
            * **w** (:obj:`numpy.ndarray`): Normal mode frequencies of the system.
            * **b** (:obj:`numpy.ndarray`): Normal mode eigenvectors of the system.
        """
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        hess_phi = ndfp.hessian()

        if "x_ep" not in self.__dict__.keys():
            self.equilibrium_position()

        hess_phi_x_ep = hess_phi(self.x_ep / self.l)

        try:
            A = np.einsum("ij,i->ij", hess_phi_x_ep, 1 / np.tile(self.m, 3))
            w, b = np.linalg.eig(A)
        except:
            A = hess_phi_x_ep / self.m
            w, b = np.linalg.eigh(A)

        w = np.sqrt(w * cst.k * cst.e ** 2 / self.l ** 3)

        _b = np.round(np.copy(b), 3).transpose()
        s = (np.sign(_b).sum(1) >= 0) * 2 - 1
        b = np.einsum("n,in->in", s, b)

        idcs = np.argsort(np.flip(w, axis=0))

        self.w, self.b, self.A = w[idcs], b[:, idcs], A
        return self.w, self.b

    def principle_axis(self, tol=1e-3):
        """
        Function that calculates the principle axes of the system.

        :Returns:
            * **x_pa** (:obj:`numpy.ndarray`): Principle axes of the system.
            * **w_pa** (:obj:`numpy.ndarray`): Normal mode frequencies of the system.
            * **b_pa** (:obj:`numpy.ndarray`): Normal mode eigenvectors of the system in the principle axis coordinate system.
        """
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
        ).astype(int)
        idcs = np.lexsort(np.concatenate(((-w_pa).reshape(1, -1), n)))

        self.x_pa, self.w_pa, self.b_pa = x_pa, w_pa[idcs], b_pa[:, idcs]
        return self.x_pa, self.w_pa, self.b_pa

    def update_params(self, **kwargs):
        """
        Updates parameters, i.e. params attribute, of a TrappedIons object.

        :Keyword Arguments:
            * **dim** (:obj:`float`): Dimension of the system.
            * **l** (:obj:`float`): Length scale of the system.
            * **m** (:obj:`float`): Mass of an ion.
            * **q** (:obj:`dict`): Charge of an ion.
        """
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
