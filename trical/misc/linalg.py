"""
Relevant linear algebra functions for TriCal
"""
import numpy as np


def norm(x):
    """
    Computes the norm of an array over the last axis
    
    Args:
        x (N-dim array of float): Array of interest
    
    Returns:
        (N-1)-dim array of float: Norm of the array of interest over the last axis
    """
    return np.sqrt((x ** 2).sum(-1))


def cartesian_to_spherical(x):
    _r = np.hypot(np.hypot(x[0], x[1]), x[2])
    _phi = np.arctan2(np.hypot(x[0], x[1]), x[2])
    _theta = np.arctan2(x[1], x[0])
    return np.array([_r, _phi, _theta])


def spherical_to_cartesian(x):
    _x = x[0] * np.sin(x[1]) * np.cos(x[2])
    _y = x[0] * np.sin(x[1]) * np.sin(x[2])
    _z = x[0] * np.cos(x[1])
    return np.array([_x, _y, _z])


def rotation_matrix(n, theta):
    n = np.einsum("ni,n->ni", n, 1 / norm(n))
    R = np.einsum("ni,nj,n->nij", n, n, 1 - np.cos(theta))
    R += np.einsum("n,ij->nij", np.cos(theta), np.identity(3))
    R[:, np.triu_indices(3, 1)[0], np.triu_indices(3, 1)[1]] += np.einsum(
        "i,ni,n->ni", np.array([-1, 1, -1]), np.flip(n, 1), np.sin(theta)
    )
    R[:, np.triu_indices(3, 1)[1], np.triu_indices(3, 1)[0]] += np.einsum(
        "i,ni,n->ni", np.array([1, -1, 1]), np.flip(n, 1), np.sin(theta)
    )
    return R


def orthonormal_subset(x, tol=1e-3):
    x = np.einsum("ni,n->ni", x, 1 / norm(x))
    i = 0
    while i < len(x):
        idcs = (
            np.logical_not(
                np.isclose(np.einsum("i,ni->n", x[i], x[i + 1 :]), 0, atol=tol)
            ).nonzero()[0]
            + i
            + 1
        )
        x = np.delete(x, idcs, axis=0)
        i += 1
    return x
