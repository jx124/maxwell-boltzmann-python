import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as mpla 
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform

class Scene:
    '''Initiates a scene containing particles with position pos, velocity vel, radius r and mass m.
    Scene dimensions dim is a (2,) array, while pos and vel are (n, 2) arrays. Simulation runs
    with timestep dt.'''

    def __init__(self, dim, pos, vel, r, m, dt):
        self.dim = dim
        self.pos = pos
        self.vel = vel
        self.r = r
        self.m = m
        self.dt = dt
        self.n = np.shape(pos)[0]
        
    def advance(self):
        '''Advance particle by time step dt.'''
        self.pos += self.vel * self.dt

        #Checks for collisions with walls
        hit_left = self.pos[:,0] < self.r 
        hit_right = self.pos[:,0] > self.dim[0] - self.r 
        hit_top = self.pos[:,1] > self.dim[1] - self.r
        hit_bottom = self.pos[:,1] < self.r

        #Updates particle position after wall collision
        self.pos[:,0] = np.where(hit_right, self.dim[0] - self.r - 2 * np.abs(self.pos[:,0] - self.dim[0] + self.r), self.pos[:,0])
        self.pos[:,0] = np.where(hit_left, self.r + 2 * np.abs(self.pos[:,0] - self.r), self.pos[:,0])
        self.pos[:,1] = np.where(hit_top, self.dim[1] - self.r - 2 * np.abs(self.pos[:,1] - self.dim[1] + self.r), self.pos[:,1])
        self.pos[:,1] = np.where(hit_bottom, self.r + 2 * np.abs(self.pos[:,1] - self.r), self.pos[:,1])

        #Flips particle velocity after wall collision
        self.vel[hit_left | hit_right, 0] *= -1
        self.vel[hit_bottom | hit_top, 1] *= -1

        #Calculates pairwise distances to check for particle collisions
        dists = squareform(pdist(self.pos))
        i, j = np.asarray(np.triu(dists <= self.r, k=1)).nonzero()
        
        #Updating particle velocties after collision
        for i, j in zip(i, j):
            v_rel = self.vel[i] - self.vel[j]
            x_rel = self.pos[i] - self.pos[j]
            delta_v = (v_rel.dot(x_rel) * x_rel) / (np.linalg.norm(x_rel) ** 2)
            self.vel[i] -= delta_v
            self.vel[j] += delta_v
            #Offset to prevent clipping 
            c = np.linalg.norm(x_rel) - 2 * self.r
            self.pos[i] -= x_rel * c/2
            self.pos[j] += x_rel * c/2

        #Updating calculations
        self.KE = 0.5 * self.m * np.sum(self.vel**2)
        self.mover2kT = (self.n * self.m)/(2 * self.KE)

#Initialising variables
n = 300                                         #number of particles
dim = (1, 1)                                    #length of box in the x and y directions
pos = np.random.random((n,2))                   #initial positions
vel = 0.5 * (np.random.random((n,2)) - 0.5)     #initial velocities
r = 0.015                                       #particle radii
m = 1                                           #paricle mass
dt = 1/30                                       #1/desired FPS
nbins = 40                                      #number of bins for histogram

#Creating simulation scene and initialising settings
scene = Scene(dim, pos, vel, r, m, dt)
fig, (sim, ax) = plt.subplots(1,2, constrained_layout = True)
sim.set_xlim(0, dim[0])
sim.set_ylim(0, dim[1])
sim.set_aspect('equal')
sim.set(xticks = [], yticks = [])

#Calculates correct scale for histogram
scene.advance()
vp = 1/np.sqrt(scene.mover2kT)
xdim = (-0.1 * vp, 5 * vp)
ydim = (0, 3 * vp * scene.mover2kT * np.exp(-1))
ax.set_aspect((xdim[1]-xdim[0])/(ydim[1]-ydim[0]))

#Calculation for theoretical distribution
boltzmannX = np.linspace(0, xdim[1], 100)
boltzmannY = 2 * scene.mover2kT * boltzmannX * np.exp(-scene.mover2kT * boltzmannX ** 2) 

#Plots
histogram = ax.hist([], bins = nbins)
approx = histogram[0]
particles, = sim.plot([],[], 'ko', markersize = 2)

def animate(i):
    scene.advance()

    #Plot particles
    particles.set_data(scene.pos[:,0],scene.pos[:,1])

    #Set axes
    ax.cla()
    ax.set_xlabel('Speed')
    ax.set_ylabel('Frequency')
    ax.set(xlim= xdim, ylim = ydim)
    
    #Plot theoretical and measured distributions
    boltzmann, = ax.plot(boltzmannX, boltzmannY, label = 'Theoretical', color = 'k')
    histogram = ax.hist(np.linalg.norm(scene.vel,axis=1), color = 'mediumblue', alpha = 0.7,
    bins = np.linspace(0, xdim[1], nbins+1), density = 1, align = 'mid')
    
    #Calculate and plot the average of the measured distribution
    global approx
    approx = approx + 2 / (i + 1) * (histogram[0] - approx)
    average, = ax.plot(np.linspace(0, xdim[1], nbins) + boltzmannX[1] / 2, approx, color = 'red', label = 'Average')

    #Create legend
    blue_patch = mpatches.Patch(color='b', alpha = 0.7, label='Measured')
    plt.legend(handles=[blue_patch, boltzmann, average])

    return particles, boltzmann, histogram, average

#Animation of 1000 frames at 30 FPS
anim = mpla.FuncAnimation(plt.gcf(), animate, 1000, interval = 1000/30)
plt.show()