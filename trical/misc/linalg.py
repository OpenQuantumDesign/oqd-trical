"""
Module containing relevant linear algebra functions for TrICal.
"""

import numpy as np


def norm(x):
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
    """
    Finds an approximate orthonormal subset of a set of vectors, after normalization.

    :param x: Set of vectors of interest.
    :type x: :obj:`numpy.ndarray`
    :param tol: Tolerance when classifying 2 vectors as orthonormal , defaults to 1e-3.
    :type tol: :obj:`float`, optional
    :returns: Orthonormal subset of the set of vectors of interest, after normalization.
    :rtype: :obj:`numpy.ndarray`
    """
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


def gram_schimdt(b, tol=1e-5):
    b = np.einsum("ij,j->ij", b, 1 / np.linalg.norm(b, axis=0))
    for i in range(b.shape[-1]):
        for j in range(i):
            b[:, i] = b[:, i] - np.dot(b[:, i], b[:, j]) * b[:, j]
            if np.linalg.norm(b[:, i]) < tol:
                b[:, i] = np.zeros(len(b[:, i]))
            else:
                b[:, i] = b[:, i] / np.linalg.norm(b[:, i])
    return b


def orthonormal_component(v, b, tol=1e-5):
    v = v / np.linalg.norm(v)
    b = gram_schimdt(b)

    v = v - np.einsum("i,ij,jk->k", v, b, b.transpose())
    if np.linalg.norm(v) < tol:
        print("Failed")
        return None
    else:
        v = v / np.linalg.norm(v)
    return v
