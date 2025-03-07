# Matrix Elements

<!-- prettier-ignore -->
/// admonition | Goal
    type: goal

Compute the multipole matrix elements with [compute_matrix_element][oqd_trical.light_matter.compiler.utils.compute_matrix_element].

///

Multipole matrix elements determine the coupling between transitions. The Rabi frequencies is defined as

$$
\Omega = \begin{cases}
    \frac{eE_0}{\hbar} \langle 1|\vec{r}_e \cdot \hat{\epsilon}|0 \rangle \, \text{for electric transitions} \\
    \frac{B_0}{\hbar} \langle 1|\vec{\mu} \cdot \hat{b}|0 \rangle \, \text{for magnetic transitions}
\end{cases}
$$

In this section we'll describe how these $\langle 1|\vec{r}_e \cdot \hat{\epsilon}|0 \rangle, \langle 1|\vec{\mu} \cdot \hat{b}|0 \rangle$ elements are computed. Currently, TrICal supports matrix element computation for:

- [Electric dipole (E1)](#__tabbed_1_1)
- [Magnetic dipole (M1)](#__tabbed_1_2)
- [Electric quadrupole (E2)](#__tabbed_1_3)

Let's consider a transition from level 0 to 1 in an ion with nuclear spin $I$ and associated quantum numbers:

| symbol          | definition                                |
| --------------- | ----------------------------------------- |
| $M_i$           | magnetization (a.k.a. $m_F$)              |
| $J_i$           | spin-orbital                              |
| $F_i$           | total angular momentum                    |
| $q = M_1 - M_0$ | change in magnetization for the transiton |

/// tab | Electric Dipole Transitions

For dipole transitions, $q$ can be $0, \pm 1$, each of which corresponds to a required polarization $\hat{q}$:

|              | $q$ | $\hat{q}$                                                     |
| ------------ | --- | ------------------------------------------------------------- |
| $\sigma_{-}$ | -1  | $\frac{1}{\sqrt{2}} \begin{pmatrix} 1\\ i\\ 0 \end{pmatrix}$  |
| $\pi$        | 0   | $\frac{1}{\sqrt{2}} \begin{pmatrix} 0\\ 0\\ 1 \end{pmatrix}$  |
| $\sigma_{+}$ | 1   | $\frac{1}{\sqrt{2}} \begin{pmatrix} 1\\ -i\\ 0 \end{pmatrix}$ |

As a result, the matrix element will depend on the overlap between the laser's polarization (electric field direction) $\hat{\epsilon}$ and the required $\hat{q}$.

<!-- prettier-ignore -->
//// admonition | Important
    type: important

////

$$
    \langle 1|\vec{r}_e \cdot \hat{\epsilon}|0 \rangle = \frac{1}{\omega_0 e}\sqrt{\frac{3\pi\epsilon_0\hbar c^3 A_{10}}{\omega_0 }} \sqrt{(2F_1 + 1)(2F_0 + 1)}
    \begin{Bmatrix}
        J_0 & J_1 & 1\\
        F_1 & F_0 & I
    \end{Bmatrix} \nonumber \\
    \sqrt{2J_1+1} \hat{q}\cdot\hat{\epsilon}\begin{pmatrix}
        F_1 & 1 & F_0\\
        M_1 & -q & -M_0
    \end{pmatrix}
$$

where $\{\}$ refers to the Wigner-6j symbol and $()$ refers to the Wigner-3j symbol.

///

/// tab | Magnetic Dipole Transitions

For dipole transitions, $q$ can be $0, \pm 1$, each of which corresponds to a required polarization $\hat{q}$:

|              | $q$ | $\hat{q}$                                                     |
| ------------ | --- | ------------------------------------------------------------- |
| $\sigma_{-}$ | -1  | $\frac{1}{\sqrt{2}} \begin{pmatrix} 1\\ i\\ 0 \end{pmatrix}$  |
| $\pi$        | 0   | $\frac{1}{\sqrt{2}} \begin{pmatrix} 0\\ 0\\ 1 \end{pmatrix}$  |
| $\sigma_{+}$ | 1   | $\frac{1}{\sqrt{2}} \begin{pmatrix} 1\\ -i\\ 0 \end{pmatrix}$ |

As a result, the matrix element will depend on the overlap between the laser's magnetic field direction $\hat{b} = \hat{k} \times \hat{\epsilon}$ and the required $\hat{q}$.

<!-- prettier-ignore -->
//// admonition | Important
    type: important

$$
    \langle 1|\vec{\mu} \cdot \hat{b}|0 \rangle = \sqrt{\frac{3 \hbar \epsilon_0 c^5 A_{10}}{16 \pi^3 \omega_0^3}} \sqrt{(2F_1 + 1)(2F_0 + 1)}
    \begin{Bmatrix}
        J_0 & J_1 & 1\\
        F_1 & F_0 & I
    \end{Bmatrix} \nonumber \\
    \sqrt{2J_1+1} \hat{q}\cdot\hat{b}\begin{pmatrix}
        F_1 & 1 & F_0\\
        M_1 & -q & -M_0
    \end{pmatrix}
$$

////

where $\{\}$ refers to the Wigner-6j symbol and $()$ refers to the Wigner-3j symbol.

///

/// tab | Electric Quadrupole Transitions

For quadrupole transitions, the laser's unit wavevector $\hat{k}$ becomes relevant for coupling (in addition to its polarization). In particular, the coupling strength is now proportional to the overlap between $\hat{k}$ and $\hat{Q}\hat{\epsilon}$.

$\hat{Q}$ is a matrix that depends on the value for $q$, which can now be one of $0, \pm 1, \pm 2$:

| $q$ | $\hat{Q}$                                                                             |
| --- | ------------------------------------------------------------------------------------- |
| -2  | $\frac{1}{\sqrt{6}}\begin{pmatrix} 1 & i & 0\\ i & -1 & 0\\ 0 & 0 & 0\end{pmatrix}$   |
| -1  | $\frac{1}{\sqrt{6}}\begin{pmatrix} 0 & 0 & 1\\ 0 & 0 & i\\ 1 & i & 0\end{pmatrix}$    |
| 0   | $\frac{1}{3}\begin{pmatrix} -1 & 0 & 0\\ 0 & -1 & 0\\ 0 & 0 & 2\end{pmatrix}$         |
| 1   | $\frac{1}{\sqrt{6}}\begin{pmatrix} 0 & 0 & -1\\ 0 & 0 & i\\ -1 & i & 0\end{pmatrix}$  |
| 2   | $\frac{1}{\sqrt{6}}\begin{pmatrix} 1 & -i & 0\\ -i & -1 & 0\\ 0 & 0 & 0\end{pmatrix}$ |

<!-- prettier-ignore -->
//// admonition | Important
    type: important

$$
    \langle 1|\vec{r}_e \cdot \hat{\epsilon}|0 \rangle = \frac{1}{\omega_0 e}\sqrt{\frac{15\pi\epsilon_0\hbar c^3 A_{10}}{\omega_0}} \sqrt{(2F_1 + 1)(2F_0 + 1)}
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
