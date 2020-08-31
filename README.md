# 2D Maxwell-Boltzmann particle simulation
A simulation of gas particles undergoing elastic collisions within a box. The speed distributions of the particles are then graphed to demonstrate that they approach the theoretical distribution (in 2 dimensions) after equilibrium is reached.
![alt text](https://wikimedia.org/api/rest_v1/media/math/render/svg/f5a7d8c97b85ecf281b42e31267014448c472cda?raw=true)

## Required python modules
Install these modules for the program to work:
* numpy
* matplotlib
* scipy

## Unresolved bugs
* Particles may stick together if they collide with very low velocity. An alternative code for offsetting the particles after collision may be required.