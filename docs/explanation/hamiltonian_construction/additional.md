# Additional Formulae

These formulas are often computed using helper functions as part of the Hamiltonian construction process. When this is the case, the name and location of the helper function is specified.

/// tab | Atom-Light Interactions

$$
    \Omega = \frac{\pi}{\tau_\pi}
$$

- $\Omega \equiv$ Rabi frequency
- $\tau_{\pi} \equiv$ $\pi$-time

<!-- prettier-ignore -->
//// admonition | Note
    type: note
Used in [set_laser_intensity_from_pi_time][trical.light_matter.interface.chamber.Chamber.set_laser_intensity_from_pi_time] and [set_laser_intensity_from_rabi_frequency][trical.light_matter.interface.chamber.Chamber.set_laser_intensity_from_rabi_frequency].

////

$$
I = \frac{\epsilon_0 c }{2}\left(\frac{\hbar\Omega}{e M_{12}}\right)^2
$$

- $M_{12} \equiv \langle 1| \hat{M}|2 \rangle \equiv$ Multipole transition matrix element

<!-- prettier-ignore -->
//// admonition | Note
    type: note
Used in [set_laser_intensity_from_pi_time][trical.light_matter.interface.chamber.Chamber.set_laser_intensity_from_pi_time] and [set_laser_intensity_from_rabi_frequency][trical.light_matter.interface.chamber.Chamber.set_laser_intensity_from_rabi_frequency].

////

$$
\eta = \vec{k}\cdot\hat{z}\sqrt{\frac{\hbar}{2m\nu}}
$$

- $\eta \equiv$ Lamb-Dicke parameter
- $\vec{k} \equiv$ Laser wavevector
- $\hat{z} \equiv$ Motional mode axis
- $m \equiv$ Ion mass
- $\nu \equiv$ Angular motional mode eigenfrequency

<!-- prettier-ignore -->
//// admonition | Note
    type: note
Implemented as [lambdicke][trical.light_matter.utilities.lambdicke]
////
///
/// tab | Lasers

$$
    E = \sqrt{\frac{2I}{\epsilon_0 c}}
$$

- $I \equiv$ Laser intensity
- $E \equiv$ magnitude of laser E-field $\vec{E}$

<!-- prettier-ignore -->
//// admonition | Note
    type: note
Used in [rabi_frequency_from_intensity][trical.light_matter.interface.chamber.Chamber.rabi_frequency_from_intensity] method
////
///
/// tab | Miscellaneous

$$
    \rho = \sum_i P_i |\psi_i\rangle
$$

- $\rho \equiv$ Density matrix
- $P_i \equiv$ Classical probability of being in state $|\psi_i \rangle$

<!-- prettier-ignore -->
//// admonition | Note
    type: note
Implemented as [to_dm][trical.light_matter.utilities.to_dm]
////
///
