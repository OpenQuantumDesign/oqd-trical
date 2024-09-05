/// tab | Chamber

## Chamber

The [Chamber][trical.light_matter.interface.chamber.Chamber] packages together the entire experimental description and takes the following arguments:

- [Chain][trical.light_matter.interface.structures.Chain] object
- List of [Laser][trical.light_matter.interface.structures.Laser] objects
- Magnetic field magnitude $B$
- Magnetic field direction $\hat{B}$

A fully assembled [Chamber][trical.light_matter.interface.chamber.Chamber] instantiation gets passed into [construct_H_tree][trical.light_matter.compiler.rule.construct_H_tree.construct_H_tree] for Hamiltonian construction.

///
/// tab | Chain

## Chain

The [Chain][trical.light_matter.interface.structures.Chain] object summarizes the dynamics of the ions trapped in a harmonic potential and takes the following arguments:

- List of [Ion][trical.light_matter.interface.ion.Ion] objects
- List of trap frequencies: [$\omega_x$, $\omega_y$, $\omega_z$]
- List of [VibrationalMode][trical.light_matter.interface.structures.VibrationalMode] objects

An [Ion][trical.light_matter.interface.ion.Ion]'s position in the list determines its ion index, which is used as its ID number.

///
/// tab | Ion

## Ion

The [Ion][trical.light_matter.interface.ion.Ion] object stores species and Hilbert-space specific information about each ion. Specifically, it keeps takes in

- Mass
- Charge
- Dictionary mapping level aliases (e.g. "S1/2") to [Level][trical.light_matter.interface.ion.Level] objects
- Dictionary of "full_transitions" (from manifold 1 to manifold 2)

Note that users do not instantiate *Ion*s directly. Instead, one would instantiate a specific ion species, say [Ca40][trical.light_matter.interface.species.Ca40], which inherits from [Ion][trical.light_matter.interface.ion.Ion]. This way, users don't have to pass in all the levels and their aliases each time they want to create a new ion.

### Level

[Level][trical.light_matter.interface.ion.Level] objects store the quantum numbers of hyperfine levels as well as their energy relative to the ground state:

- $N$: principal
- $S$: spin
- $L$: orbital
- $I$: nuclear
- $J$: spin_orbital
- $F$: spin_orbital_nuclear
- $E/h$: energy (transition frequency)

### Transition

[Transition][trical.light_matter.interface.ion.Transition] objects package together two *Level*s and store information about the Einstein A coefficient and the multipole transition type (e.g. 'E1' or 'E2'):

- [Level][trical.light_matter.interface.ion.Level] object 1
- [Level][trical.light_matter.interface.ion.Level] object 2
- $A$
- Multipole type

///
/// tab | VibrationalMode

## Vibrational Mode

Each [VibrationalMode][trical.light_matter.interface.structures.VibrationalMode] object tracks a motional degree of freedom of the ion chain, and takes the following arguments:

- Linear eigenfrequency
- Eigenvector
- Normalized motional axis
- Phonon cutoff N

///
/// tab | Laser

## Laser

<!-- prettier-ignore -->
//// admonition | In Progress
    type: warning
////

A [Laser][trical.light_matter.interface.structures.Laser] object is currently a global beam directed at all ions in the chain, though we must eventually switch over to "Beams," where we specify a [Laser][trical.light_matter.interface.structures.Laser] and [Ion][trical.light_matter.interface.ion.Ion] target. [Laser][trical.light_matter.interface.structures.Laser] objects take in the following arguments:

- Wavelength $\lambda$
- Normalized wavevector $\hat{k}$
- Intensity $I$ (in $\text{W}/\text{m}^2$)
- Phase $\phi$
- Polarization vector $\hat{\epsilon}$

///
