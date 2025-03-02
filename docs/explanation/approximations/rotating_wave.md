The rotating wave approximation (RWA) simply states that any terms in the Hamiltonian oscillating faster than a user-specified cutoff, $\omega_{\text{cutoff}}$, are neglected.

Looking at the [general Hamiltonian](../hamiltonian_construction/derivation.md#eqn:general_hamiltonian), we see that these oscillating terms will arise from the product of $e^{-i\Delta_{nmjk}t}$ and the matrix elements of $D(\alpha)$. Because we are computing these via Laguerre polynomials, it is straightforward to derive a condition on what terms must be dropped.

Let's first consider one of these oscillating terms for when $m\geq n$. Plugging in $\alpha = i\eta e^{i\nu t}$ into [first order Lamb-Dicke approximation](lamb_dicke.md#eqn:first_order_lamb_dicke), we get

$$
    \begin{align}
    e^{-i\Delta_{nmjk}t} D_{mn} &= e^{-i\Delta_{nmjk}t} \sqrt{\frac{n!}{m!}}(i\eta e^{i\nu t})^{m-n}e^{-\eta^2 /2}L_n^{(m-n)}(\eta^2)\\
    &= \sqrt{\frac{n!}{m!}}(i\eta)^{m-n}e^{i[(m-n)\nu - \Delta_{nmjk}]t- \eta^2/2}L_n^{(m-n)}(\eta^2)
    \end{align}
$$

So we find that these Hamiltonian matrix elements rotate at frequency $(m-n)\nu-\Delta_{nmjk}$. We find the same condition on this 'combined' frequency for the $m>n$ case as well

$$
    \begin{align}
    e^{-i\Delta_{nmjk}t} D_{mn} &= e^{-i\Delta_{nmjk}t} \sqrt{\frac{m!}{n!}}(-i\eta e^{-i\nu t})^{n-m}e^{-\eta^2 /2}L_m^{(n-m)}(\eta^2)\\
    &= \sqrt{\frac{m!}{n!}}(-i\eta)^{m-n}e^{-i[(n-m)\nu + \Delta_{nmjk}]t- \eta^2/2}L_m^{(n-m)}(\eta^2)
    \end{align}
$$

Thus, the condition on the RWA can be written as

$$
    \text{If  } |(m-n)\nu - \Delta_{nmkj}| < \omega_{\text{cutoff}} \text{ , } D_{mn} \rightarrow 0
$$
