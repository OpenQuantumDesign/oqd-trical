## Classes for Experiment Specification

### Chamber

The _Chamber_ packages together the entire experimental description and takes the following arguments:

- _Chain_ object
- List of _Laser_ objects
- Magnetic field magnitude $B$
- Magnetic field direction $\hat{B}$

A fully assembled _Chamber_ instantiation gets passed into _construct_H_tree.py_ for Hamiltonian construction.

### Chain

The _Chain_ object summarizes the dynamics of the ions trapped in a harmonic potential and takes the following arguments:

- List of _Ion_ objects
- List of trap frequencies: [$\omega_x$, $\omega_y$, $\omega_z$]
- List of _VibrationalMode_ objects

An _Ion_'s position in the list determines its ion index, which is used as its ID number.

### Ion

The _Ion_ object stores species and Hilbert-space specific information about each ion. Specifically, it keeps takes in

- Mass
- Charge
- Dictionary mapping level aliases (e.g. "S1/2") to _Level_ objects
- Dictionary of "full_transitions" (from manifold 1 to manifold 2)

Note that users do not instantiate *Ion*s directly. Instead, one would instantiate a specific ion species, say _Ca40_, which inherits from _Ion_. This way, users don't have to pass in all the levels and their aliases each time they want to create a new ion.

#### Level

_Level_ objects store the quantum numbers of hyperfine levels as well as their energy relative to the ground state:

- $N$: principal
- $S$: spin
- $L$: orbital
- $I$: nuclear
- $J$: spin_orbital
- $F$: spin_orbital_nuclear
- $E/h$: energy (transition frequency)

#### Transition

_Transition_ objects package together two *Level*s and store information about the Einstein A coefficient and the multipole transition type (e.g. 'E1' or 'E2'):

- _Level_ object 1
- _Level_ object 2
- $A$
- Multipole type

### VibrationalMode

Each _VibrationalMode_ object tracks a motional degree of freedom of the ion chain, and takes the following arguments:

- Linear eigenfrequency
- Eigenvector
- Normalized motional axis
- Phonon cutoff N

### Laser (IN PROGRESS)

A _Laser_ object is currently a global beam directed at all ions in the chain, though we must eventually switch over to "Beams," where we specify a _Laser_ and _Ion_ target. _Laser_ objects take in the following arguments:

- Wavelength $\lambda$
- Normalized wavevector $\hat{k}$
- Intensity $I$ (in $\text{W}/\text{m}^2$)
- Phase $\phi$
- Polarization vector $\hat{\epsilon}$
