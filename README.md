# 2D particle collision simulation

## Description
A simulation of ideal gas particles undergoing elastic collisions within a box. A histogram plot of the speed of the particles shows that it averages to the Maxwell-Boltzmann distribution for 2 dimensions after equilibrium is reached.

The speed distribution is given by:

![equation](./resources/equation.svg)

## Setup
`pip install matplotlib numpy scipy`

## Example
In this example, the particles are initially positioned in a grid with an initial velocity towards the top right + some jitter. The particles appear to compress into the corner before expanding out to fill the whole space.

![simulation](./resources/example.gif)

The accelerations of the particles are set to 0 in this simulation, but can be altered in other molecular dynamics simulations to simulate the effects of e.g. gravity, intermolecular interactions, potential fields etc. 

This simulation is inspired by similar GIF on the Wikipedia page for the ![Maxwell-Boltzmann distribution](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution)

## Version Info
* Python 3.8.0 
