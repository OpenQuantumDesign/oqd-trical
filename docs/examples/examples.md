The Hamiltonian in Equation 28, describing an N-ion (with $J_n$-levels), N-laser, L-motional mode system, is the one used to construct all Hamiltonians in TrICal.

Though it can be cumbersome to work out problems by hand for even a few ions and lasers, we know what the dynamics of simple, special cases should look like. The following are examples we work through analytically, whose results we compare with TrICal's results.

## Rabi Oscillations

We'll continue with the above system, but this time we'll keep $D(\alpha)$: coupling only to the axial mode and to first order in the Lamb-Dicke approximation (Equation 38). Equation 18 becomes

$$
    \begin{align}
    \tilde{H}_I &= \frac{\hbar\Omega}{2} \left\{e^{-i(\Delta t - \phi)}\sigma_+ \left[1 + i\eta(e^{i\nu t} a^{\dagger} + e^{-i\nu t} a)\right]\right\} + H.C.\\
    &= \frac{\hbar\Omega}{2}\left\{e^{-i(\Delta t - \phi)}\sigma_+ + i\eta e^{-i(\Delta t - \phi)}\sigma_+(e^{i\nu t} a^{\dagger} + e^{-i\nu t} a) \right\} + H.C.\\
    &= \frac{\hbar\Omega}{2}\biggl\{e^{-i(\Delta t - \phi)} + i\eta\biggr[e^{-i[(\Delta- \nu) t - \phi]}a^{\dagger}+ e^{-i[(\Delta+\nu)t - \phi]}a\biggl]\biggr\}\sigma_+  \nonumber + H.C.
    \end{align}
$$

## Sidebands

### Red Sidebands

The red sideband resonance is when $\Delta = -\nu$, in which case the above equation simplifies to:

$$
    \tilde{H}_I = \frac{\hbar\Omega}{2}\left[e^{i(\nu t + \phi)} + i\eta\left(e^{2i\nu t}a^{\dagger} + a\right)e^{i\phi}\right]\sigma_+ + H.C..
$$

Neglecting the terms oscillating at the motional mode frequency $\nu$ (or faster), we arrive at

$$
    H_{\text{RSB}} = \frac{\hbar\Omega}{2}i\eta(ae^{i\phi}\sigma_+ - a^{\dagger}e^{-i\phi}\sigma_-) + \frac{\hbar\Omega}{2}\left(e^{i\phi}\sigma_+ + e^{-i\phi}\sigma_-\right)
$$

Under this Hamiltonian, the system will also oscillate, this time between states $|1, n\rangle \leftrightarrow |2, n-1\rangle$, such that the excited state loses a phonon.

### Blue Sidebands

Lastly, if $\Delta = \nu$, we may drive the first _blue sideband_ transition

$$
    \tilde{H}_I = \frac{\hbar\Omega}{2}\left[e^{-i(\nu t - \phi)} + i\eta\left(a^{\dagger} + e^{-2i\nu t}a\right)e^{i\phi}\right]\sigma_+ + H.C.
$$

and after neglecting oscillating terms as with the RSB transition, we get

$$
    H_{\text{BSB}} = \frac{\hbar\Omega}{2}i\eta(a^{\dagger}e^{i\phi}\sigma_+ - a e^{-i\phi}\sigma_-)
$$

Now, the oscillations will result in an increased phonon number for the excited state: $|1, n\rangle \leftrightarrow |2, n+1 \rangle$
