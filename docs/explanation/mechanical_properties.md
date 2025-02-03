# Mechanical Properties of Ion Crystals

## Potential

In a system of trapped ions, the ions experience the following potentials:

$$
    \Phi = \Phi_{\mathrm{Coulomb}} + \Phi_{\mathrm{Trap}}
$$

- $\Phi_{\mathrm{Coulomb}} \equiv$ Coulomb potential.
- $\Phi_{\mathrm{Trap}} \equiv$ Trapping potential.

The trapping potential can be generated with:

- Combination of static (DC) and dynamic (RF) electric fields
- Optical fields (e.g. optical cavity, optical tweezers)

## Equilibrium Position

Calculating the equilibrium position of the ions corresponds to:

$$
\{\mathbf{r}_{i}^*\}_i^N = \mathrm{argmin}_{\{\mathbf{r}_{i}\}_i^N} \Phi\left(\{\mathbf{r}_{i}\}_i^N\right)
$$

- $\{\mathbf{r}_{i}\}_i^N \equiv$ Position of the ions.
- $\{\mathbf{r}_{i}^*\}_i^N \equiv$ Equilibrium position of the ions.

## Vibrational Modes

Calculating the vibrational (phonon) modes corresponds to:

$$
\begin{aligned}
A &= \mathrm{Hess}[\Phi]\left(\{\mathbf{r}_{i}^*\}_i^N\right) \\
A &= B^* D B
\end{aligned}
$$

- $B \equiv$ Eigenvector matrix for the ion crystal.
- $D \equiv$ Diagonal matrix containing eigenvalues for the ion crystal.
