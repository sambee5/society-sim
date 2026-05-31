import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.decomposition import PCA

# ==================================================
# PARAMETERS
# ==================================================

N = 300
d = 5
T = 500

alpha = 0.05
beta = 0.03
epsilon = 0.9
sigma = 0.01

snapshot_every = 10

# ==================================================
# INITIALIZATION
# ==================================================

np.random.seed(42)

X = np.random.uniform(-1, 1, size=(N, d))

tau = np.random.uniform(1.0, 2.0, size=N)
hysteresis = 0.2

G = nx.watts_strogatz_graph(N, k=8, p=0.1)
neighbors = [list(G.neighbors(i)) for i in range(N)]

protesting = np.zeros(N, dtype=bool)

# ==================================================
# TRACKING
# ==================================================

protest_fraction = []
avg_tensions = []

snapshots = []
protest_snapshots = []

# ==================================================
# SIMULATION
# ==================================================

for t in range(T):

    X_new = X.copy()
    step_tensions = []

    for i in range(N):

        nbrs = neighbors[i]
        if not nbrs:
            continue

        diffs = X[i] - X[nbrs]
        dists = np.linalg.norm(diffs, axis=1)

        T_i = np.mean(dists)
        step_tensions.append(T_i)

        # ------------------------------------------
        # Protest dynamics (with hysteresis)
        # ------------------------------------------

        if (not protesting[i]) and (T_i > tau[i]):
            protesting[i] = True

        elif protesting[i] and (T_i < tau[i] - hysteresis):
            protesting[i] = False

        # ------------------------------------------
        # Opinion dynamics
        # ------------------------------------------

        if protesting[i]:
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

    X = np.clip(X_new, -1, 1)

    protest_fraction.append(np.mean(protesting))
    avg_tensions.append(np.mean(step_tensions))

    if t % snapshot_every == 0:
        snapshots.append(X.copy())
        protest_snapshots.append(protesting.copy())

# ==================================================
# ================= MOBILITY METRICS =================
# ==================================================

snapshots_arr = np.array(snapshots)
num_snaps = snapshots_arr.shape[0]

cumulative_distance = np.zeros(N)
distance_over_time = np.zeros((num_snaps, N))

for t in range(1, num_snaps):
    step_dist = np.linalg.norm(
        snapshots_arr[t] - snapshots_arr[t-1],
        axis=1
    )
    cumulative_distance += step_dist
    distance_over_time[t] = cumulative_distance

# ==================================================
# ================= PLOTS =================
# ==================================================

# Protest fraction
plt.figure(figsize=(10,4))
plt.plot(protest_fraction)
plt.title("Protest Fraction Over Time")
plt.xlabel("Time")
plt.ylabel("% protesting")
plt.show()

# Tension
plt.figure(figsize=(10,4))
plt.plot(avg_tensions)
plt.title("Average Local Tension")
plt.xlabel("Time")
plt.ylabel("Tension")
plt.show()

# Mobility over time (MEAN)
plt.figure(figsize=(10,4))
plt.plot(np.mean(distance_over_time, axis=1))
plt.title("Mean Agent Mobility Over Time")
plt.xlabel("Snapshot")
plt.ylabel("Cumulative distance")
plt.show()

# Mobility distribution
plt.figure(figsize=(6,4))
plt.hist(cumulative_distance, bins=30)
plt.title("Distribution of Agent Mobility")
plt.xlabel("Total distance traveled")
plt.ylabel("Agents")
plt.show()

# Mobility vs protest (final state)
plt.figure(figsize=(6,4))
plt.scatter(cumulative_distance, protest_snapshots[-1], alpha=0.5)
plt.title("Mobility vs Protest Participation")
plt.xlabel("Total distance traveled")
plt.ylabel("Protesting (final)")
plt.show()

# ==================================================
# ================= NETWORK =================
# ==================================================

pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(6,6))
nx.draw(G, pos, node_size=40, node_color="gray", edge_color="lightgray")
plt.title("Small-World Network")
plt.show()

# ==================================================
# ================= PCA PROJECTION =================
# ==================================================

pca = PCA(n_components=2)
pca.fit(np.vstack(snapshots))

X_2d_all = [pca.transform(snap) for snap in snapshots]

# ==================================================
# ================= FINAL STATE =================
# ==================================================

X_final = X_2d_all[-1]
protest_final = protest_snapshots[-1]

plt.figure(figsize=(6,6))
plt.scatter(
    X_final[:, 0],
    X_final[:, 1],
    c=protest_final,
    cmap="coolwarm",
    alpha=0.7
)
plt.title("Final Opinion Space (Protest Highlighted)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# ==================================================
# ================= ANIMATION =================
# ==================================================

fig, ax = plt.subplots(figsize=(6,6))
scat = ax.scatter([], [], cmap="coolwarm", alpha=0.7)

all_points = np.vstack(X_2d_all)
margin = 0.5

ax.set_xlim(np.min(all_points[:,0]) - margin,
            np.max(all_points[:,0]) + margin)
ax.set_ylim(np.min(all_points[:,1]) - margin,
            np.max(all_points[:,1]) + margin)

def update(frame):
    X_2d = X_2d_all[frame]

    colors = X_2d[:, 0]
    sizes = np.where(protest_snapshots[frame], 90, 40)

    scat.set_offsets(X_2d)
    scat.set_array(colors)
    scat.set_sizes(sizes)

    ax.set_title(f"Opinion Space — t={frame * snapshot_every}")
    return scat,

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(X_2d_all),
    interval=200,
    blit=False
)

plt.show()