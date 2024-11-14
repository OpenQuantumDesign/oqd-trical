# Matrix Elements

<!-- prettier-ignore -->
/// admonition | Goal
    type: goal

Compute the multipole matrix elements with [compute_matrix_element][trical.light_matter.interface.chamber.Chamber.compute_matrix_element].

///

Multipole matrix elements determine the coupling between transitions. The Rabi frequencies is defined as $\Omega = \frac{eE_0}{\hbar} \langle 1|\vec{r}_e \cdot \hat{\epsilon}|0 \rangle$. In this section we'll describe how these $\langle 1|\vec{r}_e \cdot \hat{\epsilon}|0 \rangle$ elements are computed. Currently, TrICal supports matrix element computation for both E1 and E2 transitions.

Let's consider a transition from level 0 to 1 in an ion with nuclear spin $I$ and associated quantum numbers: $M_i, J_i, F_i$; these are the magnetization (a.k.a. $m_F$), spin-orbital, and total angular momentum quantum numbers. Also, let $q = M_2 - M_1$, the change in magnetization quantum number in the transiton.

/// tab | Dipole Transitions

For dipole transitions, $q$ can be $0, \pm 1$, each of which corresponds to a required polarization $\hat{q}$:

- $\hat{q} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1\\ i\\ 0 \end{pmatrix}$ drives $q = -1$ transitions
- $\hat{q} = \begin{pmatrix} 0\\ 0\\ 1 \end{pmatrix}$ drives $q = 0$ transitions ($\pi$ polarized)
- $\hat{q} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1\\ -i\\ 0 \end{pmatrix}$ drives $q = 1$ transitions

As a result, the matrix element will depend on the overlap between the laser's polarization $\hat{\epsilon}$ and the required $\hat{q}$.

<!-- prettier-ignore -->
//// admonition | Important
    type: important

$$
    \langle 1|\vec{r}_e \cdot \hat{\epsilon}|0 \rangle = \frac{1}{\omega_0 e}\sqrt{\frac{3\pi\epsilon_0\hbar c^3}{\omega_0 A_{10}}} \sqrt{(2F_1 + 1)(2F_0 + 1)}
    \begin{Bmatrix}
        J_0 & J_1 & 1\\
        F_1 & F_0 & I
    \end{Bmatrix} \nonumber \\
    \sqrt{2J_1+1} \hat{q}\cdot\hat{\epsilon}\begin{pmatrix}
        F_1 & 1 & F_0\\
        M_1 & -q & -M_0
    \end{pmatrix}
$$

////

where $\{\}$ refers to the Wigner-6j symbol and $()$ refers to the Wigner-3j symbol.

///

/// tab | Quadrupole Transitions

For quadrupole transitions, the laser's unit wave-vector $\hat{k}$ becomes relevant for coupling (in addition to its polarization). In particular, the coupling strength is now proportional to the overlap between $\hat{k}$ and $\hat{Q}\hat{\epsilon}$; $\hat{Q}$ is a matrix that depends on the value for $q$, which can now be one of $0, \pm 1, \pm 2$:

| $q$ | $\hat{Q}$                                                                             |
| --- | ------------------------------------------------------------------------------------- |
| -2  | $\frac{1}{\sqrt{6}}\begin{pmatrix} 1 & i & 0\\ i & -1 & 0\\ 0 & 0 & 0\end{pmatrix}$   |
| -1  | $\frac{1}{\sqrt{6}}\begin{pmatrix} 0 & 0 & 1\\ 0 & 0 & i\\ 1 & i & 0\end{pmatrix}$    |
| 0   | $\frac{1}{3}\begin{pmatrix} -1 & 0 & 0\\ 0 & -1 & 0\\ 0 & 0 & 2\end{pmatrix}$         |
| 1   | $\frac{1}{\sqrt{6}}\begin{pmatrix} 0 & 0 & -1\\ 0 & 0 & i\\ -1 & i & 0\end{pmatrix}$  |
| 2   | $\frac{1}{\sqrt{6}}\begin{pmatrix} 1 & -i & 0\\ -i & -1 & 0\\ 0 & 0 & 0\end{pmatrix}$ |

This logic is handled in TrICal under the same helper function under the _polarization_map_ dictionary.

<!-- prettier-ignore -->
//// admonition | Important
    type: important

$$
    \langle 1|\vec{r}_e \cdot \hat{\epsilon}|0 \rangle = \frac{1}{\omega_0 e}\sqrt{\frac{15\pi\epsilon_0\hbar c^3}{\omega_0 A_{10}}} \sqrt{(2F_1 + 1)(2F_0 + 1)}
    \begin{Bmatrix}
        J_0 & J_1 & 2\\
        F_1 & F_0 & I
    \end{Bmatrix} \nonumber \\  \sqrt{2J_1+1} \hat{k}\cdot\hat{Q}\hat{\epsilon}\begin{pmatrix}
        F_1 & 2 & F_0\\
        M_1 & -q & -M_0
    \end{pmatrix}
$$

////

///
