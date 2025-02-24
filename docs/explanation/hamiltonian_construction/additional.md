# Additional Formulae

These formulas are often computed using helper functions as part of the Hamiltonian construction process. When this is the case, the name and location of the helper function is specified.

<!-- prettier-ignore -->
/// admonition |
    type: card

$$
    \Omega = \frac{\pi}{\tau_\pi}
$$

- $\Omega \equiv$ Rabi frequency
- $\tau_{\pi} \equiv$ $\pi$-time

///

<!-- prettier-ignore -->
/// admonition |
    type: card

$$
I = \frac{\epsilon_0 c }{2}\left(\frac{\hbar\Omega}{e M_{12}}\right)^2
$$

- $M_{12} \equiv \langle 1| \hat{M}|2 \rangle \equiv$ Multipole transition matrix element

Used in [intensity_from_laser][oqd_trical.light_matter.compiler.utils.intensity_from_laser] and [rabi_from_intensity][oqd_trical.light_matter.compiler.utils.rabi_from_intensity]
///

<!-- prettier-ignore -->
/// admonition |
    type: card

$$
    E = \sqrt{\frac{2I}{\epsilon_0 c}}
$$

- $I \equiv$ Laser intensity
- $E \equiv$ magnitude of laser E-field $\vec{E}$

Used in [rabi_from_intensity][oqd_trical.light_matter.compiler.utils.rabi_from_intensity]
///

<!-- prettier-ignore -->
/// admonition |
    type: card

$$
\eta = \vec{k}\cdot\hat{b}\sqrt{\frac{\hbar}{2m\nu}}
$$

- $\eta \equiv$ Lamb-Dicke parameter
- $\vec{k} \equiv$ Laser wavevector
- $\hat{b} \equiv$ Motional mode eigenvector
- $m \equiv$ Ion mass
- $\nu \equiv$ Motional mode eigenfrequency

Used in [ConstructHamiltonian][oqd_trical.light_matter.compiler.codegen.ConstructHamiltonian]

///
