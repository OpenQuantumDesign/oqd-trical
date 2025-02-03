# Zeeman Shifts

<!-- prettier-ignore -->
/// admonition | Goal
    type: goal

Compute the Zeeman shift.

///

In the presence of a magnetic field, an ion's levels will be shifted, leading to a loss of degeneracy in its various manifolds. For example, the $4S_{1/2}$ manifold is two-fold degenerate because both $m_F = \frac{1}{2}$ and $m_F = -\frac{1}{2}$ have the same energy. However, when a magnetic field with magnitude $B$ is applied, the degeneracy is broken: the $m_F = \frac{1}{2}$ level will be shifted up and the $m_F = -\frac{1}{2}$ will be shifted down.

In the weak field regime, this _Zeeman_ shift is given by

$$
    \Delta_B = m_F g_F \mu_B B
$$

where $\mu_B$ is the Bohr magneton ($\approx 9.27 \times 10^{-24}$ J/T) and $g_F$ is the total angular momentum F Lande g-factor. It can be written in terms of the spin-orbital angular momentum Lande g-factor, $g_J$:

$$
    g_F = g_J \frac{F(F+1) + J(J+1) - I(I+1)}{2F(F+1)}
$$

where F, J, and I are the total, spin-orbital, and nuclear angular momentum quantum numbers, respectively. Finally,

$$
    g_J = \frac{3}{2} + \frac{S(S+1) - L(L+1)}{2J(J+1)}
$$

where $S$ is the spin angular momentum quantum number.
