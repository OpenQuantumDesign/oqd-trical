## Worked Examples

The Hamiltonian in Equation 28, describing an N-ion (with $J_n$-levels), N-laser, L-motional mode system, is the one used to construct all Hamiltonians in TrICal.

Though it can be cumbersome to work out problems by hand for even a few ions and lasers, we know what the dynamics of simple, special cases should look like. The following are examples we work through analytically, whose results we compare with TrICal's results.

### Rabi Oscillations

Let's go back to a two-level system ($|0\rangle$, $|1\rangle$) with transition frequency $\omega_0$ addressed by a single laser. We'll consider the resonant case ($\Delta = 0$) of Equation 18; let's also assume that the internal states' coupling to the motion is neglibile such that $D(\alpha) \approx \mathbb{1}$. Under these conditions, the Hamiltonian becomes time-independent:

$$
    \tilde{H}_I = \frac{\hbar\Omega}{2}\left(e^{i\phi}\sigma_+ + e^{-i\phi}\sigma_-\right)
$$

Thus, the time evolution operator is simply given by $U = \exp(-i\tilde{H}_I t/\hbar)$

$$
    U = \exp\left[-i\frac{\Omega t}{2}(e^{i\phi}\sigma_+ + e^{-i\phi}\sigma_-)\right]\\
    = \cos(\Omega t/2)I - i\sin(\Omega t/2)(e^{i\phi}\sigma_+ + e^{-i\phi}\sigma_-)\\
    = \begin{pmatrix}
        \cos(\Omega t/2 ) & -i\sin(\Omega t /2)e^{i\phi}\\
        -i\sin(\Omega t /2)e^{-i\phi} & \cos(\Omega t/2)
        \end{pmatrix}
$$

and the occupation probabilities over time for levels $|0\rangle$ and $|1\rangle$ (using the convention defined in the Section 2.1) are given by $P_0(t) = \sin^2({\Omega t /2})$ and $P_1(t) = \cos^2({\Omega t /2})$. We should expect to see complete population inversion after every $\tau = \Omega t/2$ units of times have elapsed.

### Sideband Transitions

We'll continue with the above system, but this time we'll keep $D(\alpha)$: coupling only to the axial mode and to first order in the Lamb-Dicke approximation (Equation 38). Equation 18 becomes

$$
    \tilde{H}_I = \frac{\hbar\Omega}{2} \left\{e^{-i(\Delta t - \phi)}\sigma_+ \left[1 + i\eta(e^{i\nu t} a^{\dagger} + e^{-i\nu t} a)\right]\right\} + h.c.\\
    = \frac{\hbar\Omega}{2}\left\{e^{-i(\Delta t - \phi)}\sigma_+ + i\eta e^{-i(\Delta t - \phi)}\sigma_+(e^{i\nu t} a^{\dagger} + e^{-i\nu t} a) \right\} + h.c.\\
    = \frac{\hbar\Omega}{2}\biggl\{e^{-i(\Delta t - \phi)} + i\eta\biggr[e^{-i[(\Delta- \nu) t - \phi]}a^{\dagger}+ e^{-i[(\Delta+\nu)t - \phi]}a\biggl]\biggr\}\sigma_+  \nonumber \\ &&+ h.c.
$$

In this particularly suggestive form, two more resonances are revealed: when $\Delta = \pm \nu$.

#### Red Sidebands

The red sideband resonance is when $\Delta = -\nu$, in which case the above equation simplifies to:

$$
    \tilde{H}_I = \frac{\hbar\Omega}{2}\left[e^{i(\nu t + \phi)} + i\eta\left(e^{2i\nu t}a^{\dagger} + a\right)e^{i\phi}\right]\sigma_+ + h.c.
$$

Neglecting the terms oscillating at the motional mode frequency $\nu$ (or faster), we arrive at

$$
    H_{\text{RSB}} = \frac{\hbar\Omega}{2}i\eta(ae^{i\phi}\sigma_+ - a^{\dagger}e^{-i\phi}\sigma_-) + \frac{\hbar\Omega}{2}\left(e^{i\phi}\sigma_+ + e^{-i\phi}\sigma_-\right)
  \end{equation}
Under this Hamiltonian, the system will also oscillate, this time between states $|1, n\rangle \leftrightarrow |2, n-1\rangle$, such that the excited state loses a phonon.


#### Blue Sidebands
Lastly, if $\Delta = \nu$, we may drive the first *blue sideband* transition


$$

    \tilde{H}_I = \frac{\hbar\Omega}{2}\left[e^{-i(\nu t - \phi)} + i\eta\left(a^{\dagger} + e^{-2i\nu t}a\right)e^{i\phi}\right]\sigma_+ + h.c.

$$

and after neglecting oscillating terms as with the RSB transition, we get


$$

    H_{\text{BSB}} = \frac{\hbar\Omega}{2}i\eta(a^{\dagger}e^{i\phi}\sigma_+ - a e^{-i\phi}\sigma_-)

$$

Now, the oscillations will result in an increased phonon number for the excited state: $|1, n\rangle \leftrightarrow |2, n+1 \rangle$

## What's next?
In this section, we outline a list of future features.
### Time Dependent Parameters

The Hamiltonian in Equation 28 can be further generalized to account for time-dependent parameters:


$$

    \tilde{H}_I = \sum_{n,m,j_n} \frac{\hbar\Omega_{nmjk}(t)}{2}\left[e^{-i(\Delta_{nmjk}(t) t - \phi_m(t))}\sigma_+^{(njk)} \prod_l^L D(\alpha_{nml}(t)) \right] + h.c.

$$

Note that the time dependence on $\alpha_{nml}$ could potentially come from two places: modulation of the trap potential (thereby altering the eigenmode frequencies) and modulation of the laser wavelength and/or direction (thereby changing the Lamb-Dicke parameter). Thus,


$$

    \alpha_{nml}(t) = i\eta_{nml}(t)e^{i\nu(t) t}

$$

#### Noisy Parameters
Once Hamiltonian parameters are allowed to be time-dependent, a natural extension is to allow for descriptions of noise on these and other parts of the experiment. For example, a fluctuating magnetic field will induce a fluctuating Zeeman shift, and therefore a fluctuating detuning.

### Laser $\rightarrow$ Beam
Right now, *Laser* objects are treated as global beams, such that all ions are irradiated by all beams. Of course, we'd want to add support for individual addressing for more complicated laser sequences.
### Decoherence
Right now, TrICal does not account for dechorence effects, which will require solving the Lindblad master equation (not just the Shrodinger equation):


$$

    \frac{\partial\hat{p}}{\partial t} = -\frac{i}{\hbar} [\hat{H}, \hat{p}] + \mathcal{L}(\hat{p})

$$

where $\hat{p}$ is the density operator and $\mathcal{L}$ is the Lindblad operator.
$$
