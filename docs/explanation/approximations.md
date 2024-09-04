## Approximations

### Lamb-Dicke Approximation

We saw in the derivation of the system Hamiltonian that the displacement operator on a motional mode is given by

$$
    D(\alpha) = \exp(\alpha a^{\dagger} + \alpha^* a)=\exp\left(i\eta e^{i\nu t}a^{\dagger} + i\eta e^{-i\nu t }a\right)
$$

after plugging in the definition of the coherent state parameter $\alpha$. This operator, as is, is in general difficult to simulate due to the doubly-exponentiated time-dependence on the mode frequency. However, we may retain only the first few terms in its Taylor expansion provided that certain conditions be met.

#### First Order Approximation

For the first order condition, we consider the different ways the annihilation and creation operators can act on a Fock state $|n\rangle$ \textbf{once}: $\eta a|n\rangle = \eta \sqrt{n}  |n-1\rangle $, $\eta a^{\dagger}|n\rangle = \eta\sqrt{n+1}|n+1\rangle$. Thus, the probability $P_1$ of changing the number of phonons by 1 is given by

$$
    P_1 = (\eta\sqrt{n})^2 + (\eta\sqrt{n+1})^2
$$

If $P_1$ is small, we may expand $D(\alpha)$ to first order. The equivalent, first order condition is that if

$$
    \eta^2(2n+1) << 1
$$

then

$$
    D(\alpha) \approx \mathbb{1} + \alpha a^{\dagger} + \alpha^* a = \mathbb{1} + i\eta\left(e^{i\nu t} a^{\dagger} + e^{-i\nu t}\right)
$$

#### Second Order Approximation

We can play a very similiar game with the second order condition where we consider the different ways to increase the number of phonons by 2 of a Fock state:

$$
    \eta^2 aa|n\rangle = \eta^2 \sqrt{n(n-1)}|n-2\rangle\nonumber
$$

$$
    \eta^2 a^{\dagger} a^{\dagger} |n\rangle = \eta^2 \sqrt{(n+2)(n+1)}|n+2\rangle\nonumber
$$

So if

$$
    2\eta^4(n^2+n+1)<<1
$$

then

$$
    D(\alpha) \approx \mathbb{1} + \alpha a^{\dagger} + \alpha^* a + \frac{1}{2}\left(\alpha a^{\dagger} + \alpha^* a\right)^2
$$

#### Higher Order Approximations

If none of the above conditions are met, then TrICal will expand $D(\alpha)$ out to third order:

$$
    D(\alpha) \approx \mathbb{1} + \alpha a^{\dagger} + \alpha^* a + \frac{1}{2}\left(\alpha a^{\dagger} + \alpha^* a\right)^2 + \frac{1}{6}\left(\alpha a^{\dagger} + \alpha^* a\right)^3
$$

#### Computing $D(\alpha)$ Matrix Elements

Importantly, TrICal does not compute Equations 38, 40, and 41 (convert into a matrix) directly. This is because information about the oscillating terms, particular the frequency in $e^{\pi \nu t}$, becomes difficult to retrieve when taking the RWA (described in the following section).

Instead, the order to which we must expand is used to discard $D(\alpha)$ matrix elements, computed using Laguerre polynomials. According to Glauber and Cahill, the matrix elements of $D(\alpha)$ in the Fock basis, $D_{mn} \equiv \langle m|D(\alpha)|n\rangle$, can be written as

$$
     D_{mn} = \sqrt{\frac{n!}{m!}}\alpha^{m-n}e^{-\frac{1}{2}|\alpha|^2}L_n^{(m-n)}(|\alpha|^2)
$$

for $m \geq n$ and

$$
    D_{mn} = \sqrt{\frac{m!}{n!}}(-\alpha^*)^{n-m}e^{-\frac{1}{2}|\alpha|^2}L_m^{(n-m)}(|\alpha|^2)
$$

for $m < n$. Note that these matrix elements are not computed until the abstract syntax tree representation is converted to a QuTiP-compatible object by the _QutipConversion_ rewrite rule; this rule calls the _displace_ helper function location _utilities.py_. _displace_ itself calls _D\textunderscore mn_ also located in _utilities.py_.

It's at this point that the Lamb-Dicke approximation, and the order we've determined we must expand to, take effect: if $|m-n| > $ Lamb-Dicke order, set the matrix element to 0.

Until then, information needed to compute both the Lamb-Dicke and rotating-wave approximations is stored in a _ApproxDisplacementMatrix_ object.

### Rotating Wave Approximation

The rotating wave approximation (RWA) simply states that any terms in the Hamiltonian oscillating faster than a user-specified cutoff, $\omega_{\text{cutoff}}$, are neglected. The RWA is taken at the same time as the Lamb-Dicke approximation within the _displace_ helper function in _utilities.py_.

Looking at Equation 28, we see that these oscillating terms will arise from the product of $e^{-i\Delta_{nmjk}t}$ and the matrix elements of $D(\alpha)$. Because we are computing these via Laguerre polynomials, it is straightforward to derive a condition on what terms must be dropped.

Let's first consider one of these oscillating terms for when $m\geq n$. Plugging in $\alpha = i\eta e^{i\nu t}$ into Equation 42, we get

$$
    e^{-i\Delta_{nmjk}t} D_{mn} = e^{-i\Delta_{nmjk}t} \sqrt{\frac{n!}{m!}}(i\eta e^{i\nu t})^{m-n}e^{-\eta^2 /2}L_n^{(m-n)}(\eta^2)\\
    = \sqrt{\frac{n!}{m!}}(i\eta)^{m-n}e^{i[(m-n)\nu - \Delta_{nmjk}]t- \eta^2/2}L_n^{(m-n)}(\eta^2)
$$

So we find that these Hamiltonian matrix elements rotate at frequency $(m-n)\nu-\Delta_{nmjk}$. We find the same condition on this 'combined' frequency for the $m>n$ case as well

$$
    e^{-i\Delta_{nmjk}t} D_{mn} = e^{-i\Delta_{nmjk}t} \sqrt{\frac{m!}{n!}}(-i\eta e^{-i\nu t})^{n-m}e^{-\eta^2 /2}L_m^{(n-m)}(\eta^2)\\
    = \sqrt{\frac{m!}{n!}}(-i\eta)^{m-n}e^{-i[(n-m)\nu + \Delta_{nmjk}]t- \eta^2/2}L_m^{(n-m)}(\eta^2)
$$

Thus, the condition on the RWA can be written as

$$
    \text{If  } |(m-n)\nu - \Delta_{nmkj}| < \omega_{\text{cutoff}} \text{ , } D_{mn} \rightarrow 0
$$

### Adiabatic Elimination

The adiabatic elimination step serves to ignore negligible physics that may result in long simulation run times or simulations being aborted entirely by QuTiP.

#### Single-Photon Adiabatic Elimination

A simple example of this in action is a three level system $|0\rangle, |1\rangle, |2\rangle$ in an ion being addressed by a single laser, resonant on the $|0\rangle \leftrightarrow |1\rangle$ transition. The Hamiltonian in Equation 28 predicts coupling between all pairs of levels (assuming their transitions are all allowed, hence _single-photon_).
\begin{center}
\begin{figure}[H]
\centering
\includegraphics[width=0.4\textwidth]{single*photon.png}
\caption{Caption}
\end{figure}
\end{center}
It also predicts that $\Delta*{0002}$ and $\Delta_{0012}$ will be large since the laser is far-detuned from these transitions (refer to Section 2.1 for subscript meanings). Because the probability $P$ of exciting the transition goes as $\frac{\Omega^2}{\Delta^2}$ for large $\Delta$, we expect transitions into level 2 to be extremely unlikely.

As such, a _RewriteRule_ named _PureElimination_ is tasked with traversing the Hamiltonian tree, and if $\Omega_{00ij}^2/\Delta_{00ij}^2 << $ than the user-defined threshold, it will remove all terms coupling levels $i$ and $j$ ($|i\rangle\langle j|$). This has the same effect as simply removing $|2\rangle$ from the Hilbert space entirely as far as levels 0 and 1 are concerned.

#### Raman Transitions (IN PROGRESS)

Raman transitions make use of a third level to drive transitions that cannot be driven directly. We'll consider a three level $\Lambda$ system, irradiated by two lasers, where we'd like to drive population between $|0\rangle$ and $|1\rangle$, and we'll use virtual absorption and emission of photons from $|0\rangle$ to do so:

\begin{center}
\begin{figure}[H]
\centering
\includegraphics[width=0.4\textwidth]{raman_transition.png}
\caption{Caption}
\end{figure}
\end{center}

These transitions can be difficult to simulate because they rely on a large $\Delta$ that is usually many orders of magnitude larger than the relevant simulation timescales, specifically the effective Rabi frequency between $|0\rangle$ and $|1\rangle$. This discrepancy in timescale makes QuTiP's integrator have to take small time steps to try and resolve quickly-evolving dynamics introduced by $\Delta$, resulting in very slow simulation times or calls to the integrator being aborted.

Thus, instead of simulating the full Hamiltonian in Equation 28, we must replace the parts of the tree corresponding to the $\Lambda$ system with an effective two-level Hamiltonian:

$$
    H_{\text{eff}} = \frac{\hbar\Omega_{\text{eff}}}{2}e^{-i\delta_{\text{eff}} t}|1\rangle\langle 0|D(\alpha_{\text{eff}}) + h.c
$$

where

$$
    \Omega_{\text{eff}} = \frac{\Omega_{02}\Omega_{12}}{2\Delta}\text{,    } \vec{k}_{\text{eff}} = \vec{k}_1 - \vec{k}_2
$$

which takes on the familiar form of a two-level system being addressed by a single laser with wavevector $\vec{k}_{\text{eff}}$ and detuning $\delta_{\text{eff}}$.

$\vec{k}_{\text{eff}}$ is relevant for preserving coupling to the motional modes. In Section 2.1, we used the fact that $\vec{k}\cdot\vec{r} = \eta(a^{\dagger} + a)$ to arrive at the displacement operator $D(\alpha)$, noting that $\eta$ depends on the laser's wavevector. As such, our modified displacement operator becomes:

$$
    D(\alpha_{\text{eff}})
    = \exp \left(i(\vec{k}_1 - \vec{k}_2)\cdot\vec{r}\right)\\
    = \exp \left(i(\eta_1(a^{\dagger} + a) - \eta_2(a^{\dagger} + a))\right)
$$

After performing our usual transformation into the interaction picture,

$$
    D(\alpha_{\text{eff}})
    = \exp \left(i(\eta_1(e^{i\nu t} a^{\dagger} + e^{-i\nu t}a) - \eta_2(e^{i\nu t}a^{\dagger} + e^{-i\nu t}a))\right)\\
    = \exp(\alpha_1 a^{\dagger} - \alpha_1^{*}a - \alpha_2 a^{\dagger} + \alpha_2^{*}a)\\
    = \exp\left((\alpha_1-\alpha_2)a^{\dagger} - (\alpha_1 - \alpha_2)^* a\right)\\
    = \exp\left(\alpha_{\text{eff}}a^{\dagger} - \alpha_{\text{eff}}^* a\right)
$$

so $\alpha_{\text{eff}} = i(\eta_1- \eta_2)e^{i\nu t}$.

Importantly, the three levels will be light shifted by varying amounts. Specifically, the intermediate level will be shifted toward higher energies, and the other two levels will be shifted toward lower energies:

$$
    \delta_0^{\text{LS}} = -\frac{\Omega_{02}^2}{4\Delta}
$$

$$
    \delta_1^{\text{LS}} = -\frac{\Omega_{12}^2}{4(\Delta + \delta)}
$$

$$
    \delta_2^{\text{LS}} = \frac{\Omega_{02}^2}{4\Delta} + \frac{\Omega_{12}^2}{4(\Delta + \delta)}
$$

Right now, TrICal DOES NOT account for these Stark shifts. On one hand, one can just set $\delta_{\text{eff}} = \delta + \Sigma_i\delta_i^{\text{LS}}$, which is fine if the $\Lambda$ system is isolated (coupling only exists between these three levels when the Hamiltonian is initially constructed).
