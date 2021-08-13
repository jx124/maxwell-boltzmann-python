import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from scipy.spatial.distance import pdist, squareform
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'cm'

class Particles:
    '''Initialises a scene of particles with positions pos, velocities vel, accelerations accel, radii r, and mass m.
    All of which are (n, 2) numpy arrays.'''

    def __init__(self, pos, vel, accel, r = 0.01, m = 1):
        self.pos = pos
        self.vel = vel
        self.accel = accel
        self.r = r
        self.m = m
        self.n = np.shape(pos)[0]
        
        self.KE = 0.5 * self.m * np.sum(self.vel**2)
        self.m_over_2kT = (self.n * self.m)/(2 * self.KE)

    def update(self, dt, box):
        '''Updates the positions and velocities of the particles after time step dt.'''
        self.vel += self.accel*dt
        self.pos_temp = self.pos + self.vel*dt
        
        self.sweep_and_prune() 
        self.handle_box_collision(box) 
        
        self.pos = self.pos_temp
  
    def handle_box_collision(self, box):
        '''Calculates particle positions and velocities after collision with the box.'''
        # Check if particles are outside the box
        hit_left = self.pos_temp[:,0] - self.r <= box.left
        hit_right = self.pos_temp[:,0] + self.r >= box.right
        hit_bottom = self.pos_temp[:,1] - self.r <= box.bottom
        hit_top = self.pos_temp[:,1] + self.r >= box.top

        # Adjusting positions of particles outside of box
        self.pos_temp[:,0] = np.where(hit_left, 2*(box.left + self.r) - self.pos_temp[:,0], self.pos_temp[:,0])
        self.pos_temp[:,0] = np.where(hit_right, 2*(box.right - self.r) - self.pos_temp[:,0], self.pos_temp[:,0])
        self.pos_temp[:,1] = np.where(hit_bottom, 2*(box.bottom + self.r) - self.pos_temp[:,1], self.pos_temp[:,1])
        self.pos_temp[:,1] = np.where(hit_top, 2*(box.top-self.r) - self.pos_temp[:,1], self.pos_temp[:,1])

        # Flip velocities of particles that hit the edges of the box
        self.vel[hit_left | hit_right, 0] *= -1
        self.vel[hit_bottom | hit_top, 1] *= -1

        # Rechecks if any particles are still outside the box (e.g. when particles hit near a corner)
        hit_left = self.pos_temp[:,0] - self.r <= box.left
        hit_right = self.pos_temp[:,0] + self.r >= box.right
        hit_bottom = self.pos_temp[:,1] - self.r <= box.bottom
        hit_top = self.pos_temp[:,1] + self.r >= box.top
            
        # Repeats until none of the particles are outside
        if np.any([hit_left | hit_right | hit_top | hit_bottom]):
            self.handle_box_collision(box)
        
    def sweep_and_prune(self):
        '''Implements the "Sweep and Prune" algorithm to reduce the number of pairwise distance calculations required.'''
        # Sort particles by their y coordinates
        self.sorted = np.sort(self.pos_temp[:,1])
        self.indices = np.argsort(self.pos_temp[:,1])

        flag = False
        active = set({})
        
        # For each particle, if its interval intersects with that of the next particle, add it to the active set
        for i in range(1,self.n):
            if self.sorted[i] - self.sorted[i-1] < 2 * self.r:
                active.update((self.indices[i-1],self.indices[i]))
                flag = True  
            elif flag == True:
                # When the previous particle no longer intersects, pass the active set to the collision calculations
                self.calculate_collisions(np.array(list(active)))
                active = set({})
                flag = False
            else:
                pass
            
        # Pass the last active set to the collision calculations
        self.calculate_collisions(np.array(list(active)))

    def calculate_collisions(self, active):
        '''Calculate the collisions among the active particles determined by the "Sweep and Prune" algorithm.'''
        # Check the pairwise distances between particles in the active set and return a mask of intersecting particles
        # If none are intersecting, return a mask of all False
        try:
            dists = squareform(pdist(self.pos_temp[active]))
            mask = np.transpose(np.triu(dists <= 2*self.r, k=1).nonzero())
        except:
            mask = np.ma.make_mask_none(np.shape(active))
        
        # Update positions and velocities of colliding particles
        for i, j in active[mask]:
            v_rel = self.vel[i] - self.vel[j]
            x_rel = self.pos[i] - self.pos[j]
            delta_v = (v_rel.dot(x_rel) * x_rel) / (np.linalg.norm(x_rel) ** 2)
            self.vel[i] -= delta_v
            self.vel[j] += delta_v

            c = np.linalg.norm(x_rel) - 2 * self.r
            self.pos_temp[i] -= x_rel * c/2
            self.pos_temp[j] += x_rel * c/2
            
        self.KE = 0.5 * self.m * np.sum(self.vel**2)
        self.m_over_2kT = (self.n * self.m)/(2 * self.KE)
        
class Box:
    '''Creates a sqaure box object with the given boundaries.'''
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

# Setting up the simulation by creating a square grid of particles all going in a single direction with no acceleration
x = np.linspace(0.1,0.9,20)
y = np.linspace(0.1,0.9,20)
X,Y = np.meshgrid(x,y)

positions = np.vstack([Y.ravel(), X.ravel()]).T
velocities = np.ones_like(positions) * (0.5,0.3) + np.random.randn(*np.shape(positions)) * 0.01
accels = np.zeros_like(positions) 

p = Particles(positions, velocities, accels, 0.007)
b = Box(1.0, 0.0, 0.0, 1.0)
dt = 1/100
nbins = 40

# Creating plots
fig, (sim, hist) = plt.subplots(1, 2, figsize=(10,5))

# Simulation plot settings
sim.set_xlim(b.left, b.right)
sim.set_ylim(b.bottom, b.top)
sim.set_aspect('equal')
sim.set(xticks = [], yticks = [])

# Scaling constants
v_peak = 1/np.sqrt(p.m_over_2kT)
x_dims = (-0.1 * v_peak, 5 * v_peak)
y_dims = (0, 1.1 * v_peak * p.m_over_2kT)

# Histogram plot settings
hist.set(xlim = x_dims, ylim = y_dims)
hist.set_xlabel('Speed $(m\ s^{-1})$', fontsize=12)
hist.set_ylabel('Frequency $(s\ m^{-1})$', fontsize=12)
hist.set_aspect((x_dims[1]-x_dims[0])/(y_dims[1]-y_dims[0]))

# Text box for displaying elapsed time
time_text = hist.text(0.7, 0.75, '', transform=hist.transAxes,
                      fontsize=15, bbox = dict(facecolor = 'white', edgecolor = 'black'))

# Initialise approx and calculate theoretical distribution
approx = np.zeros(nbins)
theoretical_X = np.linspace(0, x_dims[1], 100)
theoretical_Y = 2 * p.m_over_2kT * theoretical_X * np.exp(-p.m_over_2kT * theoretical_X ** 2) 

# Plot lines
ln_sim, = sim.plot([], [],c='k', linewidth=0, ms=4, marker='o')
ln_hist, = hist.plot(theoretical_X, theoretical_Y, 'k--', lw=1.5, label='Theoretical')
ln_approx, = hist.plot([], [], 'r', lw=2, label ='Average')

# Plot histogram
bins = np.linspace(0, x_dims[1], 40)
_, _, bin_container = hist.hist(np.linalg.norm(p.vel, axis=1), bins = bins,
                                density=True, color='tab:blue', label='Histogram')

plt.legend(loc='upper right')
plt.tight_layout()

# Animation code
def animate(i):
    # Update particle locations and time value
    ln_sim.set_data(p.pos[:,0], p.pos[:,1])
    time_text.set_text(f"$t={i*dt:.2f}\ s$")
    
    # Update histogram
    speeds = np.linalg.norm(p.vel, axis=1)
    n, _ = np.histogram(speeds, bins = bins, density=True)
    for count, rect in zip(n, bin_container):
        rect.set_height(count)
    
    # Update average velocity plot
    global approx
    approx = approx + 2 / (i + 1) * (np.concatenate((np.array([0]), n)) - approx)
    approx_x = np.concatenate((np.array([0]),np.linspace(0, x_dims[1], nbins-1) + x_dims[1]/((nbins-1)*2)))
    ln_approx.set_data(approx_x, approx)
    
    # Update particles
    p.update(dt, b)
    
    return ln_sim, ln_approx, rect
    
# Create and save animation
ani = animation.FuncAnimation(fig, animate, frames = 2000, interval=50, blit = True, repeat=True, repeat_delay=1000)
ani.save('particle.gif', writer = 'pillow', fps = 50, dpi = 200)