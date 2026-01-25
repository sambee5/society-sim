import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA

# --------------------
# PARAMETERS
# --------------------
N = 300          # number of agents
d = 5            # opinion dimensions
T = 500          # timesteps

alpha = 0.5     # attraction strength
beta = 0.1      # repulsion strength
epsilon = 0.9   # bounded confidence
sigma = 0.01     # noise strength

rho = 0.1        # protest critical mass
snapshot_every = 10  # timesteps between snapshots

# --------------------
# INITIALIZATION
# --------------------
np.random.seed(42)

# Opinion vectors of all agents with d opinions each in [-1, 1]
X = np.random.uniform(-1, 1, size=(N, d))

# Individual tolerance thresholds
tau = np.random.uniform(0.3, 0.6, size=N)

# Social network where each agent is connected to k neighbors with rewiring probability p
G = nx.watts_strogatz_graph(N, k=8, p=0.1)
neighbors = [list(G.neighbors(i)) for i in range(N)] #list of k neighbors for each agent
 
# Tracking
protest_fraction = []
snapshots = []
protest_snapshots = []

# --------------------
# SIMULATION LOOP
# --------------------

protesting = np.zeros(N, dtype=bool)

for t in range(T):
    #make a temp copy of X to update opinions simultaneously
    X_new = X.copy()

    for i in range(N):
        nbrs = neighbors[i]
        if not nbrs:
            continue

        # Distances to neighbors
        diffs = X[i] - X[nbrs]
        dists = np.linalg.norm(diffs, axis=1)

        # Local tension
        T_i = np.mean(dists)

        # Protest condition
        if T_i > tau[i]:
            protesting[i] = True

            # Repulsion force
            directions = diffs / (dists[:, None] + 1e-6)
            repulsion = np.mean(directions, axis=0)
            X_new[i] += beta * repulsion

        else:
            close = dists < epsilon
            if np.any(close):
                avg_opinion = np.mean(X[nbrs][close], axis=0)
                X_new[i] += alpha * (avg_opinion - X[i])

        # Noise
        X_new[i] += sigma * np.random.randn(d)

    # Clamp opinions
    X = np.clip(X_new, -1, 1)

    # Track protest size
    protest_fraction.append(np.mean(protesting))
    
    if t % snapshot_every == 0:
        snapshots.append(X.copy())
        protest_snapshots.append(protesting.copy())


# --------------------
# RESULTS
# --------------------
plt.figure(figsize=(10, 4))
plt.plot(protest_fraction)
plt.axhline(rho, color='r', linestyle='--', label='Critical mass')
plt.xlabel("Time")
plt.ylabel("Fraction protesting")
plt.title("Protest Dynamics Over Time")
plt.legend()
plt.show()



plt.figure(figsize=(6, 6))

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    node_size=40,
    node_color="gray",
    edge_color="lightgray",
    with_labels=False
)

plt.title("Watts–Strogatz Small-World Network")
plt.show()



pca = PCA(n_components=3)
pca.fit(np.vstack(snapshots)) 
X_2d_all = [pca.fit_transform(Xsnap) for Xsnap in snapshots]


X_final = X_2d_all[-1]
protest_final = protest_snapshots[-1]

plt.figure(figsize=(6, 6))
plt.scatter(
    X_final[:, 0],
    X_final[:, 1],
    c=protest_final,
    cmap="coolwarm",
    alpha=0.7
)
plt.xlabel("Opinion Axis 1")
plt.ylabel("Opinion Axis 2")
plt.title("Final Opinion Space (Red = Protesting)")
plt.colorbar(label="Protest State")
plt.show()


fig, ax = plt.subplots(figsize=(6, 6))
scat = ax.scatter([], [], c=[], cmap="coolwarm", alpha=0.7)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

def update(frame):
    # Project opinions
    X_2d = pca.transform(snapshots[frame])

    # Color by first principal component (main ideological axis)
    colors = X_2d[:, 0]

    scat.set_offsets(X_2d)
    scat.set_array(colors)

    ax.set_title(f"Opinion Space — t={frame * snapshot_every}")
    return scat,




ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(X_2d_all),
    interval=200
)





plt.show()