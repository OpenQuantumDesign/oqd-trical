
# TrICal Trapped Ion Calculator

This is a package for simulating a chain of trapped ions. The aim of this package is to remove the need for rewriting the same code to calculate equilibrium spacing, normal modes, and $J_{ij}$ every time a trap is being designed.

## Class Structure
![class structure](class_structure.png)

The basic form of the class structure is shown in the diagram. Typically in an ion trap there are three major terms in the potential; the DC part, the RF part and the coulomb part. In this code the user will input the DC and RF parts into the potential class. For the RF potential the pondermotive approximation will have to be taken before inputing the potential into the code. 

## Usage

The typical usage for this code will be to find the normal modes and frequencies for a given trap and for a given set of voltage applied to the trap electrodes. The potential around the centre of the trap is should be calculated via an appropriate method. The results of that calculation can then be input into the Potential class via three methods. The potential class is then used to calculate equilibrium spacing and normal modes. These are in turn fed into the SimulatedSpinLattice class to get out $J_{ij}$ coefficients.

## PolynomialPotential

The PolynomialPotential approximates your trapping potential in the region around the ion chain with the following multivariate polynomial:
$$
    p \,(x,y,z) = \sum_{ijk} \alpha_{ijk}  x^iy^jz^k
$$


There are three ways to effectivly input the alpha values for your potential. To demonstrate all three methods the same trapping potential with $\omega_x = 2\pi\times6$MHz, $\omega_y = 2\pi\times5$MHz, and $\omega_z = 2\pi\times1$MHz will be entered via all three methods. This potential written in full is:
$$
    V(x, y, z) = \alpha_{2,0,0}x^2 + \alpha_{0,2,0}y^2 + \alpha_{0,0,2}z^2
$$



The first is to directly input the alpha values in a $N\times N \times N$ matrix. The matrix must be the same dimention in every axis. This is shown bellow where in the p matrix the index indicates the power in $x$, $y$, or $z$.


```python
import sys

sys.path.append("../")
from trical.classes import *
from trical.constants import m_a
import numpy as np
from scipy.constants import mega, kilo, micro

mass = m_a["Yb171"]
omega_x = 2 * np.pi * 6 * mega  # Hz
omega_y = 2 * np.pi * 5 * mega  # Hz
omega_z = 2 * np.pi * 1 * mega  # Hz

# Potential in Joules
p = np.zeros((3, 3, 3))
p[2, 0, 0] = mass * (omega_x) ** 2 / 2  # alpha_2,0,0
p[0, 2, 0] = mass * (omega_y) ** 2 / 2  # alpha_0,2,0
p[0, 0, 2] = mass * (omega_z) ** 2 / 2  # alpha_0,0,2

potential_alpha = PolynomialPotential(p)
print(p)
```

    [[[0.00000000e+00 0.00000000e+00 5.60290428e-12]
      [0.00000000e+00 0.00000000e+00 0.00000000e+00]
      [1.40072607e-10 0.00000000e+00 0.00000000e+00]]
    
     [[0.00000000e+00 0.00000000e+00 0.00000000e+00]
      [0.00000000e+00 0.00000000e+00 0.00000000e+00]
      [0.00000000e+00 0.00000000e+00 0.00000000e+00]]
    
     [[2.01704554e-10 0.00000000e+00 0.00000000e+00]
      [0.00000000e+00 0.00000000e+00 0.00000000e+00]
      [0.00000000e+00 0.00000000e+00 0.00000000e+00]]]
    

The second method is to insert a symbolic expression using sympy for the potential. This is intended to be useful when you have an analyitical expression for your trapping potential. Using the same potential as the first example.


```python
import sympy as sym

x, y, z = sym.symbols("x, y ,z")
V_sym = (
    mass * (omega_x) ** 2 / 2 * x ** 2
    + mass * (omega_y) ** 2 / 2 * y ** 2
    + mass * (omega_z) ** 2 / 2 * z ** 2
)
potential_symbolic = SymbolicPotential(V_sym)
print(V_sym)
```

    2.01704554192163e-10*x**2 + 1.40072607077891e-10*y**2 + 5.60290428311563e-12*z**2
    

Finally the third method of defining a potential is to give a grid of points with the potential at each point. This grid will then be fit with polynomial and the coefficients of the polynomial will be returned in a format that can be submitted to the PolynomialPotential class via the first example method.

The format for submitting the grid to the fitting function is in two arrays. The first is a list of \[x,y,z\] positions, and the second is a list of the value of the potential at each point.


```python
rx = np.linspace(-10e-6, 10e-6, 100)  # x position grid
ry = np.linspace(-10e-6, 10e-6, 100)  # y position grid
rz = np.linspace(-40e-6, 40e-6, 100)  # z position grid

# Using the same potential funciton as before
V_lambda = sym.lambdify((x, y, z), V_sym)

# Create a gird of x, y, z points
R = np.meshgrid(rx, ry, rz)

# Get the potential value at each grid point
V_grid = V_lambda(*R)

# The potential values must be in 1D list where the position in the list
# matches the position in the coordinate list
vals = np.reshape(V_grid, (-1))

# Reshape the coordinate list to be a list of [x, y, z] positions
R = np.reshape(R, (3, -1))
R = R.transpose()
```


```python
from trical.misc import polynomial

lscale = micro  # a length scaling may have to be applied for fitting to converge

# specify the degree polynomial which has to be fitted along x, y, and z. In this case since we know the
# function is harmonic we're fitting to 2nd degree
deg = np.array([2, 2, 2])
p = polynomial.multivariate_polyfit(R, vals, deg, l=lscale)
potential_grid = PolynomialPotential(p)
print(p)
```

    [[[ 2.34755188e-34 -1.48756572e-30  5.60290428e-12]
      [-1.12038575e-28 -4.03042842e-25  4.36932156e-20]
      [ 1.40072607e-10 -1.10369821e-19 -6.07311021e-16]]
    
     [[ 1.04353603e-30  8.72044835e-27  7.60183085e-21]
      [ 3.52347123e-26 -6.73077924e-21 -3.13706099e-15]
      [-2.62240076e-22 -2.83938713e-14 -3.08082167e-10]]
    
     [[ 2.01704554e-10  1.92200332e-19 -8.65478945e-15]
      [-1.62017289e-18  2.15064378e-14  1.44870964e-08]
      [-1.55052534e-13 -2.04737035e-09  7.25693221e-05]]]
    

## TrappedIons

This class calculates the equilibrium positions, normal modes, and frequencies of a given number of ions and trapping potential. The method of inputing the potential should not have any bearing on the equilibrum position. Using the equilibrium_position funciton we can see that all three input methods give the same positions in z.


```python
N = 4

a1 = TrappedIons(N, potential_alpha, m=mass)
a2 = TrappedIons(N, potential_symbolic, m=mass)
a3 = TrappedIons(N, potential_grid, m=mass)

print("Input Alpha values Directly:", a1.equilibrium_position()[:, 2] / micro)
print("Symbolic Potential Function: ", a2.equilibrium_position()[:, 2] / micro)
print("Fitting Alpha from a Grid:   ", a3.equilibrium_position()[:, 2] / micro)
```

    Input Alpha values Directly: [-3.93817181 -1.24528026  1.24528     3.93817157]
    Symbolic Potential Function:  [-3.9381719  -1.24528033  1.24528004  3.93817162]
    Fitting Alpha from a Grid:    [-3.93817184 -1.24528027  1.24527998  3.93817154]
    

## SimulatedSpinLattice 

This class takes in a TrappedIons class and using the eigen frequencies produces a Jij coupling matrix along some axis with the axis, wavelength, detuning and adressing being input parameters.



```python
detuning = 2 * np.pi * 10 * kilo  # Hz
mu = omega_x + detuning
omega = np.ones(
    (N, 1)
)  # The intensity at each ion, this array gives uniform intensity across each ion
wavelength = 355e-9  # wavelength of light used for MS

b1 = SimulatedSpinLattice(a1, [mu], omega, k=2 * np.pi / wavelength)
```


```python
print( b1.J )
```

    [[0.00000000e+00 1.22476828e-08 1.11311619e-08 1.04014492e-08]
     [1.22476828e-08 0.00000000e+00 1.19910590e-08 1.11311619e-08]
     [1.11311619e-08 1.19910590e-08 0.00000000e+00 1.22476828e-08]
     [1.04014492e-08 1.11311619e-08 1.22476828e-08 0.00000000e+00]]
    




















