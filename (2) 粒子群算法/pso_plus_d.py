import numpy as np
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from sklearn.decomposition import PCA
import logging

# Configure logging
logging.getLogger('pyswarms').setLevel(logging.WARNING)

# Define 10-dimensional Rosenbrock function
def rosenbrock_10d(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2.0)**2.0 + (1 - x[:, :-1])**2.0, axis=1)

# Set optimization parameters
dimensions = 10
c1, c2, w = 0.5, 0.3, 0.9
n_particles = 50
iters = 1000

# Set search space boundaries
bounds = (np.array([-5]*dimensions), np.array([5]*dimensions))

# Initialize optimizer
optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, 
                          options={'c1': c1, 'c2': c2, 'w': w},
                          bounds=bounds)

# Initialize list to record best positions
best_pos_history = []

# Perform optimization and record best positions
for _ in range(iters):
    cost, pos = optimizer.optimize(rosenbrock_10d, iters=1)  # Optimize for one iteration at a time
    best_pos_history.append(optimizer.swarm.best_pos.copy())

best_pos_history = np.array(best_pos_history)

# Get particle position history
pos_history = np.array(optimizer.pos_history)
pos_history_flat = pos_history.reshape(-1, dimensions)

# PCA dimensionality reduction
pca = PCA(n_components=2)
pos_history_2d = pca.fit_transform(pos_history_flat)

# Set global style, disable LaTeX rendering
plt.rcParams['font.sans-serif'] = ['Arial']  # Use common font
plt.rcParams['axes.unicode_minus'] = False  # Properly display minus sign
plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
plt.style.use('seaborn-v0_8-deep')  # Use Seaborn style

# Plot PCA projection of particle positions
plt.figure(figsize=(10, 5))
plt.scatter(pos_history_2d[:, 0], pos_history_2d[:, 1], c=np.arange(pos_history_2d.shape[0]), cmap='viridis')
plt.colorbar(label='Iteration')
plt.title(r"PCA Projection of Particle Positions", fontsize=16, fontweight='bold')
plt.xlabel(r"$First \ Principal \ Component$", fontsize=14, fontweight='bold')
plt.ylabel(r"$Second \ Principal \ Component$", fontsize=14, fontweight='bold')
plt.grid(True)
plt.show()

# Plot best particle position change in each dimension
plt.figure(figsize=(12, 6))
for i in range(dimensions):
    plt.plot(best_pos_history[:, i], label=f'Dim {i+1}')
plt.title(r"Best Particle Position in Each Dimension", fontsize=16, fontweight='bold')
plt.xlabel(r"$Iteration$", fontsize=14, fontweight='bold')
plt.ylabel(r"$Position$", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True)
plt.show()
