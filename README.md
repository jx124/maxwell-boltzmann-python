# 2D particle collision simulation

## Description
A simulation of ideal gas particles undergoing elastic collisions within a box. A histogram plot of the speed of the particles shows that it averages to the Maxwell-Boltzmann distribution for 2 dimensions after equilibrium is reached:

## Setup
`pip install matplotlib numpy scipy`

## Example
In this example, the particles are initially positioned in a grid with an initial velocity towards the top right + some jitter.
![simulation](./resources/example.gif)
The accelerations of the particles are set to 0 in this simulation, but can be altered in other molecular dynamics simulations to simulate the effects of e.g. gravity, intermolecular interactions, potential fields etc.

## Version Info
* Python 3.8.0 
