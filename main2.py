import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ==================================================
# MODEL PARAMETERS
# ==================================================

N = 100
d = 5
T = 200

alpha = 0.05
sigma = 0.01

hysteresis = 0.2
snapshot_every = 10
protest_onset_threshold = 0.10

# ==================================================
# METRIC HELPERS
# ==================================================

def gini(values):
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        return 0.0

    values = values - np.min(values)

    total = np.sum(values)

    if total == 0:
        return 0.0

    sorted_values = np.sort(values)
    n = values.size
    index = np.arange(1, n + 1)

    return (
        2 * np.sum(index * sorted_values) / (n * total)
        - (n + 1) / n
    )


def pairwise_distances(X):
    diffs = X[:, None, :] - X[None, :, :]

    return np.linalg.norm(
        diffs,
        axis=2
    )


def fragmentation_metrics(X, epsilon):
    distances = pairwise_distances(X)

    close = distances < epsilon
    np.fill_diagonal(close, False)

    opinion_graph = nx.Graph()
    opinion_graph.add_nodes_from(range(len(X)))

    edges = np.argwhere(np.triu(close, k=1))
    opinion_graph.add_edges_from(
        (int(i), int(j)) for i, j in edges
    )

    component_sizes = np.array(
        [
            len(component)
            for component in nx.connected_components(opinion_graph)
        ],
        dtype=float
    )

    shares = component_sizes / len(X)

    return {
        "cluster_count": len(component_sizes),
        "fragmentation": 1 - np.max(shares),
        "effective_clusters": 1 / np.sum(shares**2)
    }


def safe_group_gap(values, group):
    values = np.asarray(values)
    group = np.asarray(group, dtype=bool)

    if np.any(group) and np.any(~group):
        return np.mean(values[group]) - np.mean(values[~group])

    return 0.0

# ==================================================
# SIMULATION FUNCTION
# ==================================================

def run_simulation(beta, epsilon, seed=0):

    np.random.seed(seed)

    X = np.random.uniform(-1, 1, size=(N, d))

    tau = np.random.uniform(1.0, 2.0, size=N)

    protesting = np.zeros(N, dtype=bool)

    G = nx.watts_strogatz_graph(N, k=8, p=0.1)
    neighbors = [list(G.neighbors(i)) for i in range(N)]

    protest_fraction = []
    avg_tensions = []

    mobility = np.zeros(N)
    protest_time = np.zeros(N)
    activation_count = np.zeros(N)

    for t in range(T):

        X_new = X.copy()
        previous_protesting = protesting.copy()
        step_tensions = []

        for i in range(N):

            nbrs = neighbors[i]

            if not nbrs:
                continue

            diffs = X[i] - X[nbrs]

            dists = np.linalg.norm(diffs, axis=1)

            # Normalized tension
            T_i = np.mean(dists) / np.sqrt(d)
            step_tensions.append(T_i)

            # ----------------------------------
            # Protest activation
            # ----------------------------------

            if (not protesting[i]) and (T_i > tau[i]):
                protesting[i] = True

            elif protesting[i] and (T_i < tau[i] - hysteresis):
                protesting[i] = False

            # ----------------------------------
            # Opinion update
            # ----------------------------------

            if protesting[i]:

                directions = diffs / (dists[:, None] + 1e-6)

                repulsion = np.mean(directions, axis=0)

                X_new[i] += beta * repulsion

            else:

                close = dists < epsilon

                if np.any(close):

                    avg_opinion = np.mean(
                        X[nbrs][close],
                        axis=0
                    )

                    X_new[i] += alpha * (
                        avg_opinion - X[i]
                    )

            X_new[i] += sigma * np.random.randn(d)

        mobility += np.linalg.norm(
            X_new - X,
            axis=1
        )

        activation_count += (
            (~previous_protesting) & protesting
        )

        protest_time += protesting

        X = np.clip(X_new, -1, 1)

        protest_fraction.append(
            np.mean(protesting)
        )

        avg_tensions.append(
            np.mean(step_tensions)
        )

    # ==========================================
    # RESEARCH METRICS
    # ==========================================

    final_distances = pairwise_distances(X)
    upper_triangle = np.triu_indices(N, k=1)
    pairwise = final_distances[upper_triangle]

    protest_fraction = np.array(protest_fraction)
    avg_tensions = np.array(avg_tensions)
    above_onset = np.where(
        protest_fraction >= protest_onset_threshold
    )[0]

    fragmentation = fragmentation_metrics(
        X,
        epsilon
    )

    ever_protested = protest_time > 0

    return {
        "mean_protest": np.mean(protest_fraction),
        "peak_protest": np.max(protest_fraction),
        "final_protest": protest_fraction[-1],
        "protest_onset_time": (
            above_onset[0]
            if len(above_onset) > 0
            else T
        ),
        "protest_persistence": np.mean(protest_time / T),
        "protest_volatility": np.std(np.diff(protest_fraction)),
        "activation_rate": np.mean(activation_count),
        "final_polarization": np.mean(pairwise),
        "polarization_dispersion": np.std(pairwise),
        "mean_tension": np.mean(avg_tensions),
        "final_tension": avg_tensions[-1],
        "total_mobility": np.mean(mobility),
        "mobility_inequality": gini(mobility),
        "protest_mobility_gap": safe_group_gap(
            mobility,
            ever_protested
        ),
        "cluster_count": fragmentation["cluster_count"],
        "fragmentation": fragmentation["fragmentation"],
        "effective_clusters": fragmentation["effective_clusters"]
    }

# ==================================================
# PARAMETER SWEEP
# ==================================================

epsilon_values = np.linspace(0.2, 2.0, 15)

beta_values = np.linspace(0.0, 0.05, 15)

num_runs = 5

# ==========================================
# STORAGE MATRICES
# ==========================================

protest_heatmap = np.zeros(
    (len(beta_values), len(epsilon_values))
)

polarization_heatmap = np.zeros_like(
    protest_heatmap
)

mobility_heatmap = np.zeros_like(
    protest_heatmap
)

peak_protest_heatmap = np.zeros_like(
    protest_heatmap
)

onset_heatmap = np.zeros_like(
    protest_heatmap
)

persistence_heatmap = np.zeros_like(
    protest_heatmap
)

volatility_heatmap = np.zeros_like(
    protest_heatmap
)

activation_heatmap = np.zeros_like(
    protest_heatmap
)

tension_heatmap = np.zeros_like(
    protest_heatmap
)

mobility_inequality_heatmap = np.zeros_like(
    protest_heatmap
)

protest_mobility_gap_heatmap = np.zeros_like(
    protest_heatmap
)

fragmentation_heatmap = np.zeros_like(
    protest_heatmap
)

effective_clusters_heatmap = np.zeros_like(
    protest_heatmap
)

polarization_dispersion_heatmap = np.zeros_like(
    protest_heatmap
)

# ==================================================
# SWEEP
# ==================================================

for i, beta in enumerate(beta_values):

    print(f"beta {i+1}/{len(beta_values)}")

    for j, epsilon in enumerate(epsilon_values):

        protests = []
        peak_protests = []
        onset_times = []
        persistences = []
        volatilities = []
        activation_rates = []
        polarizations = []
        polarization_dispersions = []
        tensions = []
        mobilities = []
        mobility_inequalities = []
        protest_mobility_gaps = []
        fragmentations = []
        effective_clusters = []

        for seed in range(num_runs):

            result = run_simulation(
                beta=beta,
                epsilon=epsilon,
                seed=seed
            )

            protests.append(
                result["mean_protest"]
            )

            peak_protests.append(
                result["peak_protest"]
            )

            onset_times.append(
                result["protest_onset_time"]
            )

            persistences.append(
                result["protest_persistence"]
            )

            volatilities.append(
                result["protest_volatility"]
            )

            activation_rates.append(
                result["activation_rate"]
            )

            polarizations.append(
                result["final_polarization"]
            )

            polarization_dispersions.append(
                result["polarization_dispersion"]
            )

            tensions.append(
                result["mean_tension"]
            )

            mobilities.append(
                result["total_mobility"]
            )

            mobility_inequalities.append(
                result["mobility_inequality"]
            )

            protest_mobility_gaps.append(
                result["protest_mobility_gap"]
            )

            fragmentations.append(
                result["fragmentation"]
            )

            effective_clusters.append(
                result["effective_clusters"]
            )

        protest_heatmap[i, j] = np.mean(protests)

        peak_protest_heatmap[i, j] = np.mean(
            peak_protests
        )

        onset_heatmap[i, j] = np.mean(
            onset_times
        )

        persistence_heatmap[i, j] = np.mean(
            persistences
        )

        volatility_heatmap[i, j] = np.mean(
            volatilities
        )

        activation_heatmap[i, j] = np.mean(
            activation_rates
        )

        polarization_heatmap[i, j] = np.mean(
            polarizations
        )

        polarization_dispersion_heatmap[i, j] = np.mean(
            polarization_dispersions
        )

        tension_heatmap[i, j] = np.mean(
            tensions
        )

        mobility_heatmap[i, j] = np.mean(
            mobilities
        )

        mobility_inequality_heatmap[i, j] = np.mean(
            mobility_inequalities
        )

        protest_mobility_gap_heatmap[i, j] = np.mean(
            protest_mobility_gaps
        )

        fragmentation_heatmap[i, j] = np.mean(
            fragmentations
        )

        effective_clusters_heatmap[i, j] = np.mean(
            effective_clusters
        )

# ==================================================
# PLOT FUNCTION
# ==================================================

def plot_heatmap(
    matrix,
    title,
    colorbar_label
):

    plt.figure(figsize=(8, 6))

    plt.pcolormesh(
        epsilon_values,
        beta_values,
        matrix,
        shading="auto"
    )

    plt.colorbar(
        label=colorbar_label
    )

    plt.xlabel(
        "Confidence Threshold (epsilon)"
    )

    plt.ylabel(
        "Repulsion Strength (beta)"
    )

    plt.title(title)

    plt.show()

# ==================================================
# PLOTS
# ==================================================

plot_heatmap(
    protest_heatmap,
    "Mean Protest Fraction",
    "Protest Fraction"
)

plot_heatmap(
    peak_protest_heatmap,
    "Peak Protest Fraction",
    "Peak Fraction"
)

plot_heatmap(
    onset_heatmap,
    "Protest Onset Time",
    "Time Step"
)

plot_heatmap(
    persistence_heatmap,
    "Protest Persistence",
    "Mean Fraction of Time Protesting"
)

plot_heatmap(
    volatility_heatmap,
    "Protest Volatility",
    "Std. Change in Protest Fraction"
)

plot_heatmap(
    activation_heatmap,
    "Protest Activation Rate",
    "Mean Activations per Agent"
)

plot_heatmap(
    polarization_heatmap,
    "Final Polarization",
    "Polarization"
)

plot_heatmap(
    polarization_dispersion_heatmap,
    "Polarization Dispersion",
    "Std. Pairwise Distance"
)

plot_heatmap(
    tension_heatmap,
    "Mean Local Tension",
    "Tension"
)

plot_heatmap(
    mobility_heatmap,
    "Agent Mobility",
    "Mobility"
)

plot_heatmap(
    mobility_inequality_heatmap,
    "Mobility Inequality",
    "Gini Coefficient"
)

plot_heatmap(
    protest_mobility_gap_heatmap,
    "Protest Mobility Gap",
    "Ever-Protested Minus Never-Protested Mobility"
)

plot_heatmap(
    fragmentation_heatmap,
    "Opinion Fragmentation",
    "1 - Largest Cluster Share"
)

plot_heatmap(
    effective_clusters_heatmap,
    "Effective Opinion Clusters",
    "Inverse Simpson Count"
)

# ==================================================
# OPTIONAL:
# PHASE BOUNDARY DETECTOR
# ==================================================

grad_x = np.gradient(
    protest_heatmap,
    axis=1
)

grad_y = np.gradient(
    protest_heatmap,
    axis=0
)

boundary_strength = np.sqrt(
    grad_x**2 + grad_y**2
)

plot_heatmap(
    boundary_strength,
    "Protest Phase Boundaries",
    "Gradient Magnitude"
)
