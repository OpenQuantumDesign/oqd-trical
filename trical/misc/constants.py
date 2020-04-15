"""
Module containing relevant constants, in SI units, for TrICal.

:Variables:
    * **e** (:obj:`float`): Elementary charge.
    * **hbar** (:obj:`float`): Reduced Planck constant.
    * **k** (:obj:`float`): Coulomb constant.
    * **m_a** (:obj:`dict`): Dictionary of atomic masses.
"""

e = 1.602176621e-19
hbar = 1.054571818e-34
k = 8.987551792e09
m_a = {"Ba133": 2.206956176e-25, "Yb171": 2.838464469e-25}


def convert_m_a(A):
    """
    Converts atomic mass from atomic mass units to kilograms

    :param A: Atomic mass in atomic mass units
    :type A: :obj:`float`
    :returns: Atomic mass in kilograms
    :rtype: :obj:`float`
    """
    return A * 1.66053906660e-27


def natural_l(m, q, omega):
    """
    Calculates a natural length scale for a trapped ion system

    :param m: Mass of ion
    :type m: :obj:`float`
    :param q: Charge of ion
    :type q: :obj:`float`
    :param omega: Trapping strength
    :type omega: :obj:`float`
    :returns: Natural length scale
    :rtype: :obj:`float`
    """
    return (2 * k * q ** 2 / (m * omega ** 2)) ** (1 / 3)


def natural_V(m, q, omega):
    """
    Calculates a natural energy scale for a trapped ion system

    :param m: Mass of ion
    :type m: :obj:`float`
    :param q: Charge of ion
    :type q: :obj:`float`
    :param omega: Trapping strength
    :type omega: :obj:`float`
    :returns: Natural energy scale
    :rtype: :obj:`float`
    """
    return k * q ** 2 / natural_l(m, q, omega)
