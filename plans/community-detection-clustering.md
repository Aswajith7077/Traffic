# Community Detection + Clustering Combinations — Implementation Plan

This document describes how Louvain/Leiden (graph **community detection**) and
DBSCAN (density **clustering**) are composed in both possible orders, how the
four resulting partitions are scored, and how they plug into the RL pipeline.

## 1. Background

The RL meta-policy consumes a partition of the road network: each region
(cluster) contributes one 10-dim token to the transformer, and the number of
regions `M` fixes the `SubGoalGenerator` architecture. The partition therefore
must be (a) spatially meaningful, (b) fully covering (every traffic-light node
belongs to a region), and (c) coarse enough for `M` to be tractable.

Two algorithm families produce partitions:

| Family               | Algorithms        | Unit          | Granularity |
|----------------------|-------------------|---------------|-------------|
| Community detection  | Louvain, Leiden   | graph nodes   | modularity-driven |
| Clustering           | DBSCAN            | point cloud   | density-driven |

Running them in sequence — `stage_1` then `stage_2` — yields **4 combinations**:

| Method name      | Order                          | Stage 1                 | Stage 2                    |
|------------------|--------------------------------|-------------------------|----------------------------|
| `dbscan_louvian` | clustering → community detect. | DBSCAN (fine cells)     | Louvain on supernode graph |
| `dbscan_leiden`  | clustering → community detect. | DBSCAN (fine cells)     | Leiden on supernode graph  |
| `louvian_dbscan` | community detect. → clustering | Louvain (communities)   | DBSCAN within communities  |
| `leiden_dbscan`  | community detect. → clustering | Leiden (communities)    | DBSCAN within communities  |

## 2. Methodology per combination

### 2.1 clustering → community detection (`dbscan_louvian`, `dbscan_leiden`)

1. **DBSCAN stage (fine cells).** Run DBSCAN over road-network node
   coordinates with auto-tuned `eps`/`min_samples` (see
   `services/dbscan.py:auto_tune_config`). A *fine* target mean cell size of
   4–10 nodes is used so the supernode graph is rich enough for community
   detection to merge meaningfully. Noise points are reassigned to the nearest
   cell centroid (`services/hybrid.py:_reassign_noise_labels`) so coverage
   is 100%.
2. **Contraction.** Contract cells into a *supernode graph*: one node per
   DBSCAN cell, an undirected edge between cells weighted by the number of
   road links connecting them (`services/hybrid.py:_supernode_graph`).
3. **Community-detection stage.** Run Louvain (`community_louvain.best_partition`)
   or Leiden (`leidenalg.find_partition`, `ModularityVertexPartition`) on the
   supernode graph. Each detected community groups several DBSCAN cells.
4. **Flatten.** A region = the union of the road nodes of all cells in one
   supernode community.

### 2.2 community detection → clustering (`louvian_dbscan`, `leiden_dbscan`)

1. **Community-detection stage.** Build an undirected road graph with static
   weights = number of lanes per edge, then run Louvain or Leiden. This yields
   communities (coarse regions).
2. **Clustering stage (within-community refinement).** For *each* community,
   run DBSCAN over that community's node coordinates (auto-tuned, fine target)
   to subdivide spatially spread communities into compact sub-regions. Noise is
   reassigned within the community.
3. **Flatten.** A region = one DBSCAN sub-region of one community. The final
   region count is ≥ the number of communities.

> Note on outcomes: the granularity flow differs between the two orders.
> `clustering → community detection` *merges* density cells, so compact/dense
> networks (e.g. cologne8) can collapse to a single region. The
> `community detection → clustering` order *refines* communities and never
> reduces the count below the number of communities. Both outcomes are
> legitimate and are surfaced by the metrics before training.

## 3. Evaluation metrics (community detection + clustering)

Every generated partition JSON contains a `metrics` object:

```jsonc
{
  "clusters": { "0": ["node1", "node2", ...], ... },
  "metrics": {
    "order": "louvian_then_dbscan",
    "first_stage": "louvian",
    "second_stage": "dbscan",
    "dbscan_stage": {
      "n_clusters": 10,                // DBSCAN cells / sub-regions
      "n_noise_before_reassign": 2,
      "noise_fraction_before_reassign": 0.03,
      "eps": 152.2, "min_samples": 7, "target_mean_size": "4-10",
      "silhouette": 0.19, "davies_bouldin": 0.5, "calinski_harabasz": 34.2
    },
    "cd_stage": {
      "modularity": 0.71,              // Louvain/Leiden modularity
      "n_communities": 10,
      "cd_method": "louvian",
      "cluster_quality": { "0": { "internal": 12.0, "external": 3.0, "ratio": 4.0 }, ... }
    },
    "n_clusters": 11,                  // final regions (M)
    "coverage": 1.0,
    "cluster_size_statistics": { "min": 1, "max": 20, "mean": 7.1, "median": 6.0, "gini": 0.4 },
    "silhouette": 0.19,                // final-partition clustering quality
    "davies_bouldin": 0.5,
    "calinski_harabasz": 34.2
  }
}
```

- **Community-detection metrics:** `modularity`, number of communities, and
  per-community internal/external edge weight ratio (`cluster_quality`).
- **Clustering metrics:** `silhouette`, `davies_bouldin`, `calinski_harabasz`,
  noise fraction, cluster-size statistics (computed by the shared DBSCAN
  toolkit in `services/dbscan.py`).
- **Final-partition metrics:** `n_clusters` (= `M` for the RL), `coverage`,
  size statistics, and the clustering metrics re-computed on the final labels.

## 4. Pluggable pipeline integration

The registry `region-splitting/services/registry.py` maps a method name to a
partitioner:

- `create_service(method, net_config_path, traci_service, eps, min_samples)`
  returns the configured `BaseClusteringService` for `leiden`, `louvian`,
  `dbscan`, or one of the four hybrid combinations.
- All services expose the same `build_graph()` / `get_clusters()` /
  `generate_visualization()` interface, so the pipeline treats every method
  identically.

Selection flows through three layers:

1. **CLI** — `pipeline.py --cluster-method <method>` (and
   `region-splitting/main.py --method <method>`).
2. **Env var** — `CLUSTER_METHOD` is exported by the pipeline into every
   training/eval subprocess.
3. **Config** — `advesarial/src/config.py:_resolve_cluster_path()` loads
   `clusters/<method>/<scenario>_clusters.json`, honoring the explicit method
   verbatim and falling back to `dbscan` → `leiden`.

### Commands

```bash
# Generate every partition + metrics for all 5 scenarios
cd region-splitting && python generate_clusters.py

# Full pipeline for one scenario + one combination
python pipeline.py --scenario cologne8 --cluster-method louvian_dbscan

# Train + evaluate cologne8 only (dedicated script)
python train_eval_cologne8.py --cluster-method louvian_dbscan --episodes 10 --episode-steps 1000 --steps 500
```

## 5. Training, evaluation and SUMO run

1. **Train** `advesarial/src/sample.py` (REINFORCE-style, 4 Adam optimizers)
   for the selected method. Training metrics (meta / actor-critic / subgoal
   losses, global reward) are written to `advesarial/metrics.txt` and plotted
   under `advesarial/models/<scenario>/run_<ts>/plots/`.
2. **Evaluate** `advesarial/src/evaluate.py` — runs SUMO headless for a fixed
   number of steps, executes the trained policy greedily, and reports:
   - **Average Travel Time** (s)
   - **Average Delay Time** (s) = travel time − free-flow travel time
   - plus queue, waiting-time and completed-vehicle statistics.
3. Optionally play the model back live in the SUMO GUI:
   `cd advesarial && python run_gui.py --scenario cologne8 --steps 500`.

## 6. Files touched / added

| File | Change |
|------|--------|
| `region-splitting/services/registry.py` | new — pluggable method → service registry |
| `region-splitting/services/hybrid.py` | new — 4 composed combinations + metrics |
| `region-splitting/services/dbscan.py` | `auto_tune_config`, `min_samples_for`, `choose_eps`, `cluster_bounds` |
| `region-splitting/main.py` | dispatch through the registry, hybrid + auto-tuned DBSCAN |
| `region-splitting/generate_clusters.py` | new — bulk generation for all methods × scenarios |
| `region-splitting/generate_dbscan_clusters.py` | reuse shared auto-tune helpers |
| `pipeline.py` | `--cluster-method`, `--steps`, grid4x4 support, `CLUSTER_METHOD` env |
| `train_eval_cologne8.py` | new — train + evaluate cologne8 only |
| `advesarial/src/config.py` | generic `CLUSTER_METHOD` path resolution |