# Trapped-ion System Hamiltonian Derivation

<!-- prettier-ignore -->
/// admonition | Goal
    type: goal

Compute the general hamiltonian of the trapped-ion system for:

- $N$ ions
  - $J$ considered electronic states each
- $M$ lasers
- $L$ phonon modes

Implemented with [construct_H_tree][trical.light_matter.compiler.rule.construct_H_tree.construct_H_tree].

///

## Simple 2-level Ion System

We'll start by deriving the Hamiltonian for a simple, 2-level system separated by energy $\hbar\omega_0$, where $\omega_0$ is the transition frequency. When the ion is irradiated by a laser, its hydrogen-like Hamiltonian with electronic and motional degrees of freedom $H_{\text{el}} + H_{\text{mot}}$, is perturbed by $H_{ED}$, the dipole operator

$$
\begin{align}
    H &= H_{\text{el}} + H_{\text{mot}} + H_{ED} \\
    &= \frac{\hbar\omega_0}{2}\left(|1\rangle\langle1| - |0\rangle\langle 0|\right) + \hbar\nu a^{\dagger}a + H_{ED}
    \end{align}
$$

where $\nu$ is the ion's center-of-mass motion mode frequency in the harmonic potential.

To first order

$$
H_{ED} = -\vec{d}\cdot\vec{E} = e \vec{r}_e \cdot E_0\hat{\epsilon}\cos({\vec{k}\cdot\vec{r} - \omega t + \phi})
$$

- $\hat{\epsilon}$ is the laser's polarization
- $\vec{r}$ and $\vec{r}_e$ are the center of mass positions of the nucleus and electron, respectively
- $\omega$ and $\phi$ are the laser's frequency and initial phase, respectively.

Because $\vec{r}_e$ has odd parity, the diagonal entries in the dipole operator cancel, thus we can write $\vec{d}\cdot\hat{\epsilon}$ in terms of the Pauli raising and lower operators

$$
\begin{align}
    \sigma_+ &= \frac{1}{2} (\sigma_x + i \sigma_y) = |1\rangle\langle0| =
    \begin{pmatrix}
    0 & 1\\
    0 & 0
    \end{pmatrix} \\
    \sigma_- &= \frac{1}{2} (\sigma_x - i \sigma_y) = |0\rangle\langle1| =
    \begin{pmatrix}
    0 & 0\\
    1 & 0
    \end{pmatrix}
    \end{align}
$$

We'll follow the convention that the highest level in the number basis corresponds to the unit vector whose first entry is 1:

$$
|1\rangle = \begin{pmatrix}
    1\\
    0
\end{pmatrix}, \quad
|0\rangle = \begin{pmatrix}
    0\\
    1
\end{pmatrix}
$$

Making these substitutions, expanding the cosine, and taking the Rabi frequency to be $\Omega = \frac{eE_0}{\hbar} \langle 1|\vec{r}_e \cdot \hat{\epsilon}_l | 0\rangle$, we arrive a

$$
H_{ED} = \frac{\hbar\Omega}{2}(\sigma_+ + \sigma_-)\left[e^{i(\vec{k}\cdot\vec{r}-\omega t +\phi)} + e^{-i(\vec{k}\cdot\vec{r}-\omega t +\phi)}\right]
$$

<!-- prettier-ignore -->
/// admonition | Note
    type: note
The computation of the matrix elements: $\langle 1|\vec{r}_e \cdot \vec{\epsilon}_l | 0\rangle$, for dipole and quadropole transitions is described [here](/explanation/hamiltonian_construction/matrix_elements).
///

### Interaction Picture for Spin

In principle, this Hamiltonian fully describes the motional and internal dynamics of the 2-level system. However, for computational purposes and for gaining greater intuition about the system, we boost into the frame rotating at the transition frequency $\omega_0$:

$$
\begin{align}
H_0 &= H_{\mathrm{el}} = \frac{\hbar\omega_0}{2}\sigma_z, \quad U = \exp(-iH_0t/\hbar), \quad H_I = H_{ED} \\
H'_I &= U^{\dagger}H_I U \\
&= \frac{\hbar\Omega}{2} e^{i\frac{\omega_0t}{2}\sigma_z}(\sigma_+ + \sigma_-)e^{-i\frac{\omega_0t}{2}\sigma_z}[\dots] \\
&= \frac{\hbar\Omega}{2}(e^{i\omega_0t}\sigma_+ + e^{-i\omega_0t}\sigma_-)\left[e^{i(\vec{k}\cdot\vec{r}-\omega t +\phi)} + e^{-i(\vec{k}\cdot\vec{r}-\omega t +\phi)}\right]
\end{align}
$$

<!-- prettier-ignore -->
/// admonition | Note
    type: note
The transform is equivalent to: $\sigma_+ \rightarrow e^{i\omega_0 t}\sigma_+$ and $\sigma_- \rightarrow e^{-i\omega_0 t}\sigma_-$.
///

### Motional Coupling

Next, we include the effects of motional coupling by taking $\vec{k}\cdot\vec{r} = \eta(a^{\dagger} + a )$, where $a, a^{\dagger}$ are the harmonic oscillator ladder operators and $\eta$ is the Lamb-Dicke parameter

$$
    \begin{align}
    H'_I &= \frac{\hbar\Omega}{2}(e^{i\omega_0t}\sigma_+ + e^{-i\omega_0t}\sigma_-)\left\{e^{i[\eta(a^{\dagger}+a)-\omega t +\phi]} + e^{-i[\eta(a^{\dagger} + a)-\omega t +\phi}\right\}\\
    &= (\dots) \left[e^{i\eta(a^{\dagger} + a)}e^{-i(\omega t - \phi)} + e^{-i\eta(a^{\dagger} + a)}e^{i(\omega t - \phi)} \right]\\
    &= \frac{\hbar\Omega}{2} \biggl[e^{i\eta(a^{\dagger} + a)}e^{-i[(\omega-\omega_0)t - \phi]}\sigma_+ + e^{-i\eta(a^{\dagger} + a)}e^{i[(\omega+\omega_0)t - \phi]} \sigma_+ + e^{i\eta(a^{\dagger} + a)}e^{-i[(\omega+\omega_0)t - \phi]}\sigma_- + e^{-i\eta(a^{\dagger} + a)}e^{i[(\omega-\omega_0)t - \phi]} \sigma_- \biggr]
    \end{align}
$$

### Rotating Wave Approximation (RWA)

Under the rotating wave approximation (RWA), we neglect the fast oscillating terms, those with frequency $\omega_0 + \omega$, and define the detuning from resonance $\Delta = \omega - \omega_0$

$$
    H'_I = \frac{\hbar\Omega}{2} \biggl[e^{i\eta(a^{\dagger} + a)}e^{-i(\Delta t - \phi)}\sigma_+ + e^{-i\eta(a^{\dagger} + a)}e^{i(\Delta t - \phi)} \sigma_- \biggr]
$$

### Interaction Picture for Phonons

At this point we apply one more transformation: boosting into the frame rotating at the ion's motional frequency $\nu$.

$$
    \begin{align}
    H''_I &= \frac{\hbar\Omega}{2} e^{i\nu t a^{\dagger}a}\left[e^{i\eta(a^{\dagger} + a)}e^{-i(\Delta t - \phi)} + e^{-i\eta(a^{\dagger} + a)}e^{i(\Delta t - \phi)} \right]e^{-i\nu t a^{\dagger}a}
    \end{align}
$$

<!-- prettier-ignore -->
/// admonition | Note
    type: note
The transform is equivalent to $e^{i\eta(a^{\dagger} + a)} \rightarrow D(\alpha) = \exp\left(\alpha \hat{a}^{\dagger} - \alpha^* \hat{a}\right)$ where $\alpha = i\eta e^{i \nu t}$.
///

$$
    \begin{align}
    H''_I &= \frac{\hbar\Omega}{2} \biggl[e^{-i(\Delta t - \phi)}\sigma_+ D(\alpha) + e^{i(\Delta t - \phi)}\sigma_- D(-\alpha)\biggr]\\
    &= \frac{\hbar\Omega}{2} \biggl[e^{-i(\Delta t - \phi)}\sigma_+ D(\alpha)\biggr] + H.C.
    \end{align}
$$

### Additional Phonon Modes

We assumed motion in only one dimension, but we can easily generalize to $L$ mode coupling

$$
    \vec{k} \cdot \vec{r} = \sum_l \eta_l(\hat{a}_l^{\dagger} + \hat{a}_l)
$$

Because $a^{\dagger}_l, a_l$ commute with $a^{\dagger}_k,a_k$ ($l\neq k$), and $D(\alpha)$ commutes with all other terms in the Hamiltonian, we can simply write the motional term as the product of displacement operators:

<a name="eqn:single_ion_hamiltonian"></a>

<!-- prettier-ignore -->
/// admonition | Important
    type: important

$$
    H''_I = \frac{\hbar\Omega}{2}\left[e^{-i(\Delta t - \phi)}\sigma_+ \prod_l^L D(\alpha_l) \right] + H.C.
$$

///

where the coherent state parameter picks up a mode index $\alpha_l = i\eta_l e^{i \nu_l t}$. We know that for a single ion, the $l$ indices correspond to motion along $\hat{e}_x$, $\hat{e}_y$ and $\hat{e}_z$.

## Generalization

Let's now fully generalize the [Single-ion Hamiltonian](#eqn:single_ion_hamiltonian) for:

- $N$ ions
  - $J$ considered electronic states each
- $M$ lasers

### Additional Lasers

Introduce $M$ laser fields irradiating the ion. Moving forward, $m$ will represent the laser index, which marks all laser-specific parameters ($\phi_m$, $\omega_m$, $\vec{k}_m$, $\hat{\epsilon}_m$, $E_{0,m}$). $H_{ED}$ now becomes:

$$
    H_{ED} = -\vec{d}\cdot\vec{E} = e \vec{r}_e \cdot \sum_m E_{0,m} \hat{\epsilon}_m \cos(\vec{k}_m\cdot\vec{r} - \omega_m t + \phi_m)
$$

The Rabi frequency becomes $\Omega_m = \frac{eE_{0,m}}{\hbar}\langle1|\vec{r}_e\cdot\hat{\epsilon_m}|0\rangle$ as it gains laser index dependence.

$$
H_{ED} =
\sum_m\frac{\hbar\Omega_m}{2}(\sigma_+ + \sigma_-)\left[e^{i\left(\vec{k}_m\cdot\vec{r}-\omega_m t -\phi_m\right)} + e^{-i\left(\vec{k}_m\cdot\vec{r}-\omega_m t -\phi_m\right)}\right]
$$

Performing the same steps as before, we obtain:

<!-- prettier-ignore -->
/// admonition | Important
    type: important

$$
    H'_I = \sum_m^M \frac{\hbar\Omega_m}{2}\left[e^{-i(\Delta_m t - \phi_m)}\sigma_+ \prod_l^L D(\alpha_{ml}) \right] + H.C.
$$

///

where $\Delta_m = \omega_m - \omega_0$ and $\alpha_{ml} = i\eta_{ml} e^{i \nu_l t}$. The Lamb-Dicke parameter gains a laser index as it's a function of the laser's alignment with the motional mode's axis.

### Additional Ions

We now consider $N$ ions, each of which is still a 2-level system, irradiated by the superimposed classical field from $M$-lasers. Moving forward, $n$ will be the ion index.

$$
    H_{ED} = \sum_n-\vec{d}_n\cdot\vec{E}_m(\vec{r}_n, t) =\sum_n e \vec{r}_{e,n} \cdot \sum_m E_{0,m} \hat{\epsilon}_m \cos(\vec{k}_m\cdot\vec{r}_n - \omega_m t + \phi_m)
$$

Taking $\Omega_{nm} = \frac{eE_{0,m}}{\hbar}\langle1|\vec{r}_{e,n}\cdot\hat{\epsilon}_m|0\rangle$, we see the Rabi frequency, raising and lowering operators all pick up an ion index.

$$
H_{ED} =
\sum_{n,m}\frac{\hbar\Omega_{nm}}{2}\left(\sigma_+^{(n)} + \sigma_-^{(n)}\right)\left[e^{i\left(\vec{k}_m\cdot\vec{r}_n-\omega_m t +\phi_m\right)} + e^{-i\left(\vec{k}_m\cdot\vec{r}_n-\omega_m t +\phi_m\right)}\right]
$$

Performing the same steps as before, we obtain:

<!-- prettier-ignore -->
/// admonition | Important
    type: important

$$
    H'_I = \sum_n^N \sum_{m}^N \frac{\hbar\Omega_{nm}}{2}\left[e^{-i(\Delta_{nm} t - \phi_m)}\sigma_+^{(n)} \prod_l^L D(\alpha_{nml}) \right] + H.C.
$$

///

where $\Delta_{nm} = \omega_m - \omega_{0,n}$ and $\alpha_{nml} = i\eta_{nml} e^{i \nu_l t}$. The Lamb-Dicke parameter depends on the ion's mass.

### Additional Levels

Finally we consider ions where we include more than two levels in their Hilbert space. Immediately, we must modify $H_{\mathrm{el}}$ to reflect the eigen-energies associated with our $J_n$ internal states (in the $n^{\mathrm{th}}$ ion)

$$
    H_{\text{int}}^{(n)} =  \hbar\sum_{j_n}^{J_n}\omega_{0 j_n}|j_n\rangle\langle j_n|
$$

$\hbar\omega_{0 j_n}$ is the energy separation between the ground and $j_n^{\mathrm{th}}$ level. Thus, $\hbar\omega_{00} = 0$. $H_{ED}$ must also be modified to account for coupling between all level pairs; coupling is ultimately determined from their respective matrix elements.

At this point, it becomes clear that the electronic ladder operators we've been using are now ambiguous: between which level pairs does it act? Thus, we introduce some notation: let $\sigma_{+}^{(njk)} = |k_n\rangle\langle j_n|$ be the raising operator for the $|j_n\rangle \leftrightarrow |k_n\rangle$ transition in the $n^{\mathrm{th}}$ ion, where we take $k_n$ to be higher in energy than $j_n$ (the corresponding lowering operator is simply the Hermitian conjugate). Thus, $H_{ED}$ for any such level pairs in ion $n$, addressed by laser $m$, become:

$$
    H_{ED}^{(nmjk)} =
    \frac{\hbar\Omega_{nmjk}}{2}\left(\sigma_+^{(njk)} + \sigma_-^{(njk)}\right)\left[e^{i(\vec{k}_m\cdot\vec{r}_n-\omega_m t -\phi_m)}  + e^{-i(\vec{k}_m\cdot\vec{r}_n-\omega_m t -\phi_m)}\right]
$$

and the full perturbative term is now:

$$
    H_{ED} = \sum_{n}\sum_{m}\sum_{j_n\neq k_n}H_{ED}^{(nmjk)}
$$

for **_unique_** $j_n, k_n$ pairs.

Performing the same steps as before, we obtain:

<a name="eqn:general_hamiltonian"></a>

<!-- prettier-ignore -->
/// admonition | Important
    type: important

$$
H'_I = \sum_n^N \sum_{m}^N \sum_{j_n \neq k_n}^{J_n} \frac{\hbar\Omega_{nmjk}}{2}\left[e^{-i(\Delta_{nmjk} t - \phi_m)}\sigma_+^{(njk)} \prod_l^L D(\alpha_{nml}) \right] + H.C.
$$

///

where $\Delta_{nmjk} = \omega_m - \omega_{njk}$ and $\omega_{njk}$ is the transition frequency for $|j_n\rangle \leftrightarrow |k_n\rangle$ in the $n^{\mathrm{th}}$ ion.
