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

num_runs = 30

# ==================================================
# CORE SIMULATION FUNCTION
# ==================================================

def run_simulation(seed=0):

    np.random.seed(seed)

    X = np.random.uniform(-1, 1, size=(N, d))
    tau = np.random.uniform(1.0, 2.0, size=N)

    hysteresis = 0.2

    G = nx.watts_strogatz_graph(N, k=8, p=0.1)
    neighbors = [list(G.neighbors(i)) for i in range(N)]

    protesting = np.zeros(N, dtype=bool)

    protest_fraction = []
    mobility = np.zeros(N)

    snapshots = []

    for t in range(T):

        X_new = X.copy()

        for i in range(N):

            nbrs = neighbors[i]
            if not nbrs:
                continue

            diffs = X[i] - X[nbrs]
            dists = np.linalg.norm(diffs, axis=1)

            T_i = np.mean(dists)

            # -----------------------------
            # Protest dynamics (hysteresis)
            # -----------------------------
            if (not protesting[i]) and (T_i > tau[i]):
                protesting[i] = True
            elif protesting[i] and (T_i < tau[i] - hysteresis):
                protesting[i] = False

            # -----------------------------
            # Opinion update
            # -----------------------------
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

        # -----------------------------
        # Mobility (distance traveled)
        # -----------------------------
        mobility += np.linalg.norm(X_new - X, axis=1)

        X = np.clip(X_new, -1, 1)

        protest_fraction.append(np.mean(protesting))

        if t % snapshot_every == 0:
            snapshots.append(X.copy())

    # -----------------------------
    # FINAL METRICS
    # -----------------------------

    final_polarization = np.mean([
        np.linalg.norm(X[i] - X[j])
        for i in range(N) for j in range(i+1, N)
    ])

    return {
        "final_polarization": final_polarization,
        "mean_protest": np.mean(protest_fraction),
        "total_mobility": np.mean(mobility),
        "final_protest": np.mean(protesting),
        "snapshots": snapshots
    }

# ==================================================
# MONTE CARLO RUNS
# ==================================================

results = {
    "final_polarization": [],
    "mean_protest": [],
    "total_mobility": [],
    "final_protest": []
}

all_snapshots = []

for seed in range(num_runs):
    res = run_simulation(seed)

    for k in results:
        results[k].append(res[k])

    all_snapshots.append(res["snapshots"])

# flatten snapshots for PCA
flat_snapshots = np.vstack([np.vstack(s) for s in all_snapshots])

# ==================================================
# STATISTICS
# ==================================================

print("\n=== AGGREGATE RESULTS ===\n")

for k, v in results.items():
    v = np.array(v)
    print(f"{k}: mean={np.mean(v):.4f}, std={np.std(v):.4f}")

# ==================================================
# HISTOGRAMS
# ==================================================

plt.figure(figsize=(12,8))

for i, k in enumerate(results.keys()):
    plt.subplot(2,2,i+1)
    plt.hist(results[k], bins=15, alpha=0.8)
    plt.title(k)

plt.tight_layout()
plt.show()

# ==================================================
# PCA VISUALIZATION (FINAL RUN ONLY FOR CLARITY)
# ==================================================

final_snapshots = all_snapshots[-1]

pca = PCA(n_components=2)
pca.fit(np.vstack(final_snapshots))
X_2d_all = [pca.transform(s) for s in final_snapshots]

protest_final = np.zeros(N)  # optional (not tracked per snapshot here)

X_final = X_2d_all[-1]

plt.figure(figsize=(6,6))
plt.scatter(
    X_final[:,0],
    X_final[:,1],
    c="blue",
    alpha=0.6
)
plt.title("Final Opinion Space (One Run)")
plt.show()

# ==================================================
# ANIMATION (single run)
# ==================================================

fig, ax = plt.subplots(figsize=(6,6))
scat = ax.scatter([], [], alpha=0.7)

all_points = np.vstack(X_2d_all)
margin = 0.5

ax.set_xlim(np.min(all_points[:,0]) - margin,
            np.max(all_points[:,0]) + margin)
ax.set_ylim(np.min(all_points[:,1]) - margin,
            np.max(all_points[:,1]) + margin)

def update(frame):
    X_2d = X_2d_all[frame]

    scat.set_offsets(X_2d)
    scat.set_array(X_2d[:,0])

    ax.set_title(f"Opinion Space t={frame}")
    return scat,

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(X_2d_all),
    interval=200,
    blit=False
)

plt.show()