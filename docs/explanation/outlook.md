In this section, we outline a list of future features.

## Time Dependent Parameters

The Hamiltonian in Equation 28 can be further generalized to account for time-dependent parameters:

$$
    \tilde{H}_I = \sum_{n,m,j_n} \frac{\hbar\Omega_{nmjk}(t)}{2}\left[e^{-i(\Delta_{nmjk}(t) t - \phi_m(t))}\sigma_+^{(njk)} \prod_l^L D(\alpha_{nml}(t)) \right] + H.C.
$$

Note that the time dependence on $\alpha_{nml}$ could potentially come from two places: modulation of the trap potential (thereby altering the eigenmode frequencies) and modulation of the laser wavelength and/or direction (thereby changing the Lamb-Dicke parameter). Thus,

$$
    \alpha_{nml}(t) = i\eta_{nml}(t)e^{i\nu(t) t}
$$

### Noisy Parameters

Once Hamiltonian parameters are allowed to be time-dependent, a natural extension is to allow for descriptions of noise on these and other parts of the experiment. For example, a fluctuating magnetic field will induce a fluctuating Zeeman shift, and therefore a fluctuating detuning.

## Laser $\rightarrow$ Beam

Right now, _Laser_ objects are treated as global beams, such that all ions are irradiated by all beams. Of course, we'd want to add support for individual addressing for more complicated laser sequences.

## Decoherence

Right now, TrICal does not account for dechorence effects, which will require solving the Lindblad master equation (not just the Shrodinger equation):

$$
    \frac{\partial\hat{p}}{\partial t} = -\frac{i}{\hbar} [\hat{H}, \hat{p}] + \mathcal{L}(\hat{p})
$$

where $\hat{p}$ is the density operator and $\mathcal{L}$ is the Lindblad operator.
