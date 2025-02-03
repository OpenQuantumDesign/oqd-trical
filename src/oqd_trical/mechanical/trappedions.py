import numpy as np

########################################################################################

from .base import Base
from ..misc import constants as cst
from ..misc.linalg import orthonormal_subset
from ..misc.optimize import dflt_opt
from .potential import CoulombPotential

########################################################################################


class TrappedIons(Base):
    """
    Object representing a system of trapped ions.

    Args:
        N (int): Number of ions.
        ps (Potential): Potentials on the system.

    Keyword Args:
        dim (int): Dimension of system.
        l (float): Length scale of system.
        m (float): Mass of ions
        q (float): Charge of ions

    """

    def __init__(self, N, *ps, **kwargs):
        super(TrappedIons, self).__init__()

        params = {
            "dim": ps[0].dim if "dim" in ps[0].__dict__.keys() else 3,
            "l": 1e-6,
            "m": cst.convert_m_a(171),
            "q": cst.e,
        }
        params.update(kwargs)
        self.__dict__.update(params)

        self.N = N
        self.cp = CoulombPotential(N, dim=self.dim, q=self.q)

        for p in ps:
            p.update_params(N=N)
        self.ps = np.array(ps)

        self.fp = self.cp + self.ps.sum()
        pass

    def equilibrium_position(self, opt=dflt_opt, **kwargs):
        """
        Function that calculates the equilibrium position of the ions.

        Args:
            opt (Callable): Generator of the appropriate optimization function for finding the equilibrium position.

        Returns:
            x_ep (np.ndarray[float]): Equilibrium position of the ions.
        """
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])
        ndfp = ndcp + ndps.sum()

        def _ndfp(x):
            return ndfp(x.reshape(self.dim, self.N).transpose())

        self.x_ep = (
            opt(self, **kwargs)(_ndfp).reshape(self.dim, self.N).transpose() * self.l
        )
        return self.x_ep

    def normal_modes(self, block_sort=False):
        """
        Function that calculates the normal modes of the system.

        Args:
            block_sort (bool): Indicator to apply block sorting.

        Returns:
            w (np.ndarray[float]): Normal mode frequencies of the system.
            b (np.ndarray[float]): Normal mode eigenvectors of the system.
        """
        ndcp = self.cp.nondimensionalize(self.l)
        ndps = np.array([p.nondimensionalize(self.l) for p in self.ps])

        def hess_phi(x):
            return np.array([ndp.hessian()(x) for ndp in np.append(ndps, ndcp)]).sum(0)

        if "x_ep" not in self.__dict__.keys():
            self.equilibrium_position()

        hess_phi_x_ep = hess_phi(self.x_ep / self.l)

        if isinstance(self.m, float):
            A = hess_phi_x_ep / self.m
            w, b = np.linalg.eigh(A)
        else:
            A = np.einsum(
                "ij,i,j->ij",
                hess_phi_x_ep,
                1 / np.tile(np.sqrt(self.m), 3),
                1 / np.tile(np.sqrt(self.m), 3),
            )
            w, b = np.linalg.eigh(A)
            b = np.einsum("im,i->im", b, 1 / np.tile(np.sqrt(self.m), 3))
            b = b / np.linalg.norm(b, axis=0)

        w = np.sqrt(w * cst.k_e * cst.e**2 / self.l**3)

        _b = np.round(np.copy(b), 3).transpose()
        s = (np.sign(_b).sum(1) >= 0) * 2 - 1
        b = np.einsum("n,in->in", s, b)

        if block_sort:
            n = np.round(
                np.array(
                    [
                        np.linalg.norm(
                            b[i * self.N : (i + 1) * self.N].transpose(), axis=-1
                        )
                        for i in range(self.dim)
                    ]
                )
            ).astype(int)
            idcs = np.lexsort(np.concatenate(((-w).reshape(1, -1), n)))
        else:
            idcs = np.argsort(np.flip(w, axis=0))

        self.w, self.b, self.A = (
            w[idcs],
            b[:, idcs],
            A * cst.k_e * cst.e**2 / self.l**3,
        )
        return self.w, self.b

    def principal_axes(self, tol=1e-3):
        """
        Function that calculates the principle axes of the system.

        Args:
            tol (float): Tolerance for evaluating orthogonality of principal axes.

        Returns:
            x_pa (np.ndarray[float]): Principle axes of the system.
            w_pa (np.ndarray[float]): Normal mode frequencies of the system.
            b_pa (np.ndarray[float]): Normal mode eigenvectors of the system in the principal axes coordinate system.
        """
        if np.isin(np.array(["w", "b"]), np.array(self.__dict__.keys())).sum() != 2:
            self.normal_modes()

        xs = self.b.reshape(self.dim, -1).transpose()
        x_pa = orthonormal_subset(xs, tol=tol)

        assert len(x_pa) == self.dim, "Principle axes do not exist"

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
                    np.linalg.norm(
                        b_pa[i * self.N : (i + 1) * self.N].transpose(), axis=-1
                    )
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

        Args:
            dim (float): Dimension of the system.
            l (float): Length scale of the system.
            m (float): Mass of an ion.
            q (float: Charge of an ion.
        """
        self.params.update(kwargs)
        self.__dict__.update(self.params)
        pass

    def mode_ion_coupling(self):
        if (
            np.isin(
                np.array(["w_pa", "b_pa", "x_pa"]), np.array(self.__dict__.keys())
            ).sum()
            != 3
        ):
            self.principal_axes()

        mic = np.zeros((3 * self.N, 3 * self.N, 3 * self.N))
        idcs = np.triu_indices(3 * self.N, k=1)
        mic[:, idcs[0], idcs[1]] = mic[:, idcs[1], idcs[0]] = (
            self.b_pa[np.array(idcs), :].prod(axis=0).transpose()
        )
        self.mic = mic
        return self.mic

    pass
