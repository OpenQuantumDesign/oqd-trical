We saw in the derivation of the system Hamiltonian that the displacement operator on a motional mode is given by

$$
    D(\alpha) = \exp(\alpha a^{\dagger} + \alpha^* a)=\exp\left(i\eta e^{i\nu t}a^{\dagger} + i\eta e^{-i\nu t }a\right)
$$

after plugging in the definition of the coherent state parameter $\alpha$. This operator, as is, is in general difficult to simulate due to the doubly-exponentiated time-dependence on the mode frequency. However, we may retain only the first few terms in its Taylor expansion provided that certain conditions be met.

## First Order Approximation

For the first order condition, we consider the different ways the annihilation and creation operators can act on a Fock state $|n\rangle$ **once**: $\eta a|n\rangle = \eta \sqrt{n}  |n-1\rangle$, $\eta a^{\dagger}|n\rangle = \eta\sqrt{n+1}|n+1\rangle$. Thus, the probability $P_1$ of changing the number of phonons by 1 is given by

$$
    P_1 = (\eta\sqrt{n})^2 + (\eta\sqrt{n+1})^2
$$

If $P_1$ is small, we may expand $D(\alpha)$ to first order as follows:

<a name="eqn:first_order_lamb_dicke"></a>

<!-- prettier-ignore -->
/// admonition | Important
    type: important
If:

$$
    \eta^2(2n+1) << 1
$$

Then:

$$
    D(\alpha) \approx \mathbb{1} + \alpha a^{\dagger} + \alpha^* a = \mathbb{1} + i\eta\left(e^{i\nu t} a^{\dagger} + e^{-i\nu t} a\right)
$$

///

## Second Order Approximation

We can play a very similiar game with the second order condition where we consider the different ways to increase the number of phonons by 2 of a Fock state:

$$
    \begin{align}
    \eta^2 aa|n\rangle &= \eta^2 \sqrt{n(n-1)}|n-2\rangle \\
    \eta^2 a^{\dagger} a^{\dagger} |n\rangle &= \eta^2 \sqrt{(n+2)(n+1)}|n+2\rangle
    \end{align}
$$

Similar to above, we may expand $D(\alpha)$ to second order as follows:

<a name="eqn:second_order_lamb_dicke"></a>

<!-- prettier-ignore -->
/// admonition | Important
    type: important
If:

$$
    2\eta^4(n^2+n+1)<<1
$$

Then:

$$
    D(\alpha) \approx \mathbb{1} + \alpha a^{\dagger} + \alpha^* a + \frac{1}{2}\left(\alpha a^{\dagger} + \alpha^* a\right)^2
$$

///

## Higher Order Approximations

If none of the above conditions are met, then TrICal will expand $D(\alpha)$ out to third order:

$$
    D(\alpha) \approx \mathbb{1} + \alpha a^{\dagger} + \alpha^* a + \frac{1}{2}\left(\alpha a^{\dagger} + \alpha^* a\right)^2 + \frac{1}{6}\left(\alpha a^{\dagger} + \alpha^* a\right)^3
$$

## Computing $D(\alpha)$ Matrix Elements

Importantly, TrICal does not compute [first](#eqn:first_order_lamb_dicke) and [second](#eqn:second_order_lamb_dicke) order constriants (convert into a matrix) directly. This is because information about the oscillating terms, particular the frequency in $e^{\pi \nu t}$, becomes difficult to retrieve when taking the RWA (described in the following section).

Instead, the order to which we must expand is used to discard $D(\alpha)$ matrix elements, computed using Laguerre polynomials. According to Glauber and Cahill, the matrix elements of $D(\alpha)$ in the Fock basis, $D_{mn} \equiv \langle m|D(\alpha)|n\rangle$, can be written as

For $m \geq n$:

$$
     D_{mn} = \sqrt{\frac{n!}{m!}}\alpha^{m-n}e^{-\frac{1}{2}|\alpha|^2}L_n^{(m-n)}(|\alpha|^2)
$$

For $m < n$:

$$
    D_{mn} = \sqrt{\frac{m!}{n!}}(-\alpha^*)^{n-m}e^{-\frac{1}{2}|\alpha|^2}L_m^{(n-m)}(|\alpha|^2)
$$

<!-- prettier-ignore -->
/// admonition | Note
    type: note
The matrix elements are not computed until the abstract syntax tree representation is converted to a QuTiP-compatible object by the [QutipCodeGeneration][oqd_trical.backend.qutip.codegen.QutipCodeGeneration] conversion rule.

///

It's at this point that the Lamb-Dicke approximation, and the order we've determined we must expand to, takes effect: if $|m-n| >$ Lamb-Dicke order, set the matrix element to 0.
