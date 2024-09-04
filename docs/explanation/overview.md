TrICal (Trapped-Ion Calculator) is a classical simulation tool using QuTiP that, when given a description of a trapped-ion experiment...

- $N$ ions in a chain, each with $J_N$ specified levels
- $M$ lasers
- $L$ motional modes

... constructs, approximates, and evolves a system Hamiltonian. TrICal is just one part of a much larger project: QuantumION.

QuantumION is a collaborative effort between Professors Rajibul Islam, Crystal Senko, and Roger Melko working toward a full-stack quantum computer using trapped $^{133}\text{Ba}^{+}$ ions. _Full stack_ refers to the various allowable levels of abstraction in the quantum computer's interface:

![](../figures/stack_diagram.png)

At the digital layer, for example, a user would specify a quantum circuit, which gets compiled down through the stack until it reaches the trapped-ion system. There is a classical emulator at each layer; thus, TrICal is the atomic physics layer's classical output.

TrICal will receive an experimental description from the atomic layer's interpreter, which should provide the following information:

- Ions
  - Species
  - Levels
  - Motional modes (Ion chain)
- Lasers
  - Polarization $\hat{\epsilon}$
  - Wavevector $\vec{k}$
  - Intensity $I$
  - Phase $\phi$
- Chamber magnetic field $\vec{B}$
- Trap potential (COM mode frequencies in $\hat{x}$, $\hat{y}$, $\hat{z}$)

From these specifications, TrICal, constructs an initial Hamiltonian tree object, which is later traversed and mutated when performing approximations. Once all approximations are complete, the tree is converted in a QuTiP compatible object for simulation and visualization.

![](../figures/pipeline.png)

The purpose of this document is to provide context beyond the docstrings currently in TrICal, with explanations about design choices and where equations come from.

Please contact avaldesm@mit.edu with questions, comments, or concerns, about work on the light-matter interaction components of TrICal.
