# Goals

Detailed explanation of the three core components of the hierarchical RL traffic signal control system.

---

## 1. Reward Function

### Formula

The total reward at each timestep is:

```
R_total = 0.4 · R_eff + 0.2 · R_fair + 0.3 · R_emergency + 0.1 · R_emission
```

Where each component is:

```
R_eff = -β₁·ΔQ - β₂·ΔW + L_avg

R_fair = -0.3·Var(W) - 0.25·max(W - 90, 0) - 0.15·(1 - J) - 0.5·envy - 0.5·max(W_ped) - 1.2·C_ped

R_emergency = -W_emergency

R_emission = -S_total
```

### Variables

| Symbol | Meaning | Source |
|--------|---------|--------|
| ΔQ | Change in total queue length (prev - curr) | `environment.py:137` |
| ΔW | Change in total waiting time (prev - curr) | `environment.py:138` |
| L_avg | Average per-intersection local reward | `environment.py:110,134` |
| Var(W) | Variance of non-emergency vehicle wait times | `environment.py:145` |
| J | Jain's fairness index over wait times | `environment.py:148` |
| envy | max(other utilities) - my utility | `environment.py:150-151` |
| C_ped | Pedestrian-vehicle conflict count | `environment.py:129` |
| W_emergency | Emergency vehicle waiting time | `environment.py:114` |
| S_total | Total vehicle stop count (emission proxy) | `environment.py:113` |
| β₁, β₂ | Efficiency weights (default 0.1 each) | `environment.py:26-27` |

### Reasoning for Each Component

**R_eff (40% — Efficiency):** The primary goal. `ΔQ` and `ΔW` measure whether congestion is *improving* compared to the previous step — this gives the agent credit for reducing jams, not just maintaining low ones. `L_avg` is the per-intersection throughput reward (vehicles served), ensuring intersections don't just minimize queue by doing nothing. The sign convention is positive when queue/wait decreases.

**R_fair (20% — Fairness):** Prevents the agent from optimizing one corridor at the expense of others:
- `Var(W)` — penalizes unequal wait distribution across vehicles
- `max(W - 90, 0)` — hard penalty when any vehicle waits >90s (prevents starvation)
- `1 - J` — Jain's index measures how evenly wait times are distributed (J=1 is perfectly fair)
- `envy` — the difference between the best-served intersection's utility and the agent's own, forcing the sub-policy to not ignore its own intersection
- `max(W_ped)` and `C_ped` — pedestrian wait and conflict penalties prevent vehicle-only optimization

**R_emergency (30% — Emergency):** High weight because emergency vehicles (ambulances, fire trucks) have hard priority requirements. Any waiting time for EMVs is penalized directly.

**R_emission (10% — Emission):** Vehicle stops are a proxy for emissions (acceleration/deceleration cycles produce more pollution than steady driving). Lower stops = smoother flow = less emissions.

### Why These Specific Weights?

The 0.4/0.2/0.3/0.1 split reflects priority: efficiency and emergency response matter most in traffic control. Fairness prevents pathological solutions where some roads are permanently green while others are permanently red. Emissions are secondary but included to incentivize smooth flow.

The reward is then normalized via exponential moving average (`_normalize_reward` at `environment.py:180-185`) to stabilize training — raw rewards vary wildly in magnitude across episodes.

---

## 2. How the Meta-Policy Affects the Sub-Policy

The hierarchical structure has a clear **top-down information flow** with **gradient isolation**:

### Architecture

```
                    META-POLICY (Global)
                    ====================
Cluster states ──→ TransformerEncoder ──→ global_embedding + subregion_embeddings
                         │                        │              │
                         │                        │              ▼
                         │                        │      SubGoalGenerator (LSTM)
                         │                        │              │
                         │                        │              ▼
                         │                        │        G_t (goal vector)
                         │                        │         [W_part | Q_part]
                         │                        │
                    SUB-POLICY (Local)             │
                    ====================           │
Raw observations ──→ LocalEncoder ──→ GATLayer ──→│ concat with mean ──→ ActorCritic
                                                          ↑
                                                    G_t.detach()  ←── gradient stop
```

### Concrete Data Flow

**Step 1 — Meta-policy produces a goal:**

```python
def _find_global_observation():          # sample.py:110
    cluster_states = traci_service.get_cluster_states(clusters)  # (M, 10) per cluster
    global_encoding, local_encoding = transformer_encoder(cluster_states.unsqueeze(0))
    subgoal_vector = subgoal_generator(local_encoding, global_encoding)  # G_t
    return subgoal_vector
```

The TransformerEncoder (`transformer.py:35-63`) processes M cluster-level 10-dim states. It prepends a learnable `global_token` (like a CLS token), runs self-attention, then splits into:
- `global_embedding` (the CLS token output — network-wide summary)
- `subregion_embeddings` (per-cluster representations)

The SubGoalGenerator (`lstm.py:24-37`) runs an LSTM over the M subregion embeddings, flattens them, concatenates with the global encoding, and produces a goal vector `G_t` of dimension `2 * num_intersections`.

**Step 2 — Sub-policy uses the goal as context:**

```python
def _find_local_observations():          # sample.py:89
    observations = traci_service.get_observations()   # (N, 10) per intersection
    hidden_state = local_encoder(observations)        # (N, 64) per intersection
    local_features = GAT(hidden_state, adjacency_list) # (N, 64) with neighbor info
    final_state = [cat(z_i, mean(features)) for z_i in local_features]  # (N, 128)
    return final_state
```

Each intersection's 10-dim observation is encoded to 64-dim, then the GAT aggregates neighbor information via graph attention. The final state is the concatenation of local features with the network-wide mean (128-dim total).

**Step 3 — ActorCritic acts, goal alignment trains:**

```python
action_prob, state_values = actor_critic(final_state)  # (N, 7) phases per intersection

# During training (sample.py:171-181):
ac_loss, _ = compute_ac_loss(actor_critic, states, actions, rewards, next_states, dones, gamma)
sub_loss = compute_goal_alignment_loss(w_global, q_global, subgoal_vector.detach(), beta1, beta2)
total_loss = ac_loss + eta2 * sub_loss
```

### How the Goal Actually Influences Sub-Policy Behavior

The goal `G_t` doesn't directly change the sub-policy's forward pass — it influences it through **two training signals**:

1. **Goal alignment loss** (`loss.py:52-65`): The goal vector is split in half: `G_w` (first half, predicting global waiting) and `G_q` (second half, predicting global queue). The loss penalizes when these predictions diverge from actual global W and Q. This trains the meta-policy to produce goals that encode the true network state, making the goal a *useful representation*.

2. **Gradient isolation** (`sample.py:177`): `subgoal_vector.detach()` ensures the meta-policy's gradients don't flow back through the sub-policy's loss. The two levels train independently but share the same goal representation.

In essence, the meta-policy learns to compress the global traffic state into a compact goal vector, and the sub-policy learns to use its local observations + GAT neighbor info to make phase decisions that are *globally aware* rather than purely local.

---

## 3. Automatic Region Splitting

### The Problem with Manual Splitting

Traditional approaches (including the HiLight paper) require domain experts to manually partition the traffic network into regions based on:
- Geographic proximity
- Administrative boundaries
- Intuition about traffic flow corridors
- Manual tuning of region sizes

This is:
- **Non-reproducible** — different experts produce different partitions
- **Network-specific** — doesn't generalize to new cities
- **Static** — doesn't adapt to changing traffic patterns
- **Labor-intensive** — doesn't scale to large networks

### What We Implemented

The `region-splitting/` module replaces manual partitioning with **two graph-based community detection algorithms** that use live traffic data from SUMO. The pipeline runs as a standalone step before RL training and produces cluster JSON files consumed by the adversarial module.

#### Pipeline Overview

```
SUMO Network (.net.xml)
    │
    ▼
sumolib reads network topology ──→ NetworkX/igraph graph
    │
    ▼
TraciService runs 1000 simulation steps
    │
    ▼
Dynamic edge weights computed per step
    │
    ▼
Community detection (Louvain or Leiden)
    │
    ▼
Post-processing (merge small clusters)
    │
    ▼
clusters/{algorithm}/{network}_clusters.json
```

#### Algorithm 1: Louvain (`services/louvian.py`)

**Library:** `python-louvain` (NetworkX-based)

**Step 1 — Build directed graph from SUMO** (`louvian.py:23-36`):

```python
def build_graph(self):
    for edge in self.net.getEdges():
        if edge.isSpecial():
            continue
        source = edge.getFromNode().getID()
        destination = edge.getToNode().getID()
        self.graph.add_edge(source, destination, id=edge.getID(),
                           static_weight=edge.getLaneNumber())
```

Each intersection becomes a node, each road becomes a directed edge. The `static_weight` stores lane count as a fallback.

**Step 2 — Compute dynamic edge weights** (`louvian.py:38-54`):

```python
def compute_weights(self, max_iter=1000):
    edge_weights = {}
    for _ in range(max_iter):
        self.traci_service.step()
        for edge in self.net.getEdges():
            if edge.isSpecial():
                continue
            weight = self.traci_service.get_edge_weight(edge)
            edge_weights[edge.getID()] = edge_weights.get(edge.getID(), 0) + weight
    for key in edge_weights:
        edge_weights[key] /= max_iter
```

Runs 1000 SUMO simulation steps and accumulates per-edge weights. The weight formula (`region-splitting/services/traci.py:103-105`):

```
weight = 1.0 · vehicle_count + 0.3 · waiting_time + 2.0 · congestion_ratio · vehicle_count
```

This captures three traffic signals: how many vehicles are on the road, how long they're waiting, and how congested the road is relative to its capacity. Averaging over 1000 steps smooths out transient spikes.

**Step 3 — Convert to weighted undirected graph** (`louvian.py:61-76`):

Louvain requires an undirected graph. Directed edges (A→B and B→A) are merged by summing their weights.

**Step 4 — Run Louvain** (`louvian.py:78-101`):

```python
partition = community_louvain.best_partition(undirected_graph, weight="weight")
modularity = community_louvain.modularity(partition, undirected_graph, weight="weight")
```

Louvain greedily optimizes modularity Q — the fraction of edges within communities minus the expected fraction if edges were random. It iterates: moving nodes between communities to maximize Q, then aggregating communities into super-nodes, until no improvement is possible.

**Step 5 — Merge small clusters** (`louvian.py:125-153`):

Clusters with fewer than `min_cluster_size=5` nodes are merged into their best neighboring large cluster (highest edge weight connection). This prevents trivially small regions.

**Status:** Implemented but currently **commented out** in `main.py:44`. The `compute_weights()` method is defined but never called from `build_graph()`, so dynamic weights aren't actually used — only static lane counts are applied. This is a known bug.

---

#### Algorithm 2: Leiden (`services/leiden.py`) — Active

**Library:** `leidenalg` + `python-igraph`

**Step 1 — Compute dynamic weights first** (`leiden.py:48-64`):

Unlike Louvain, Leiden calls `compute_weights()` inside `build_graph()`, so dynamic traffic-based weights are actually used.

**Step 2 — Build igraph from SUMO** (`leiden.py:15-46`):

```python
def build_graph(self):
    self.compute_weights()
    vertices = set()
    edges_to_add = []
    for edge in self.net.getEdges():
        if edge.isSpecial():
            continue
        source = edge.getFromNode().getID()
        destination = edge.getToNode().getID()
        vertices.add(source)
        vertices.add(destination)
        weight = self.edge_weights.get(edge_id, 0)
        edges_to_add.append((source, destination, weight))
    self.graph.add_vertices(list(vertices))
    for source, destination, weight in edges_to_add:
        self.graph.add_edge(source, destination, weight=weight)
```

**Step 3 — Run Leiden** (`leiden.py:66-84`):

```python
partition = leidenalg.find_partition(
    self.graph,
    leidenalg.ModularityVertexPartition,
    weights="weight",
)
```

Leiden is an improvement over Louvain with three key differences:
1. **Faster** — uses a different local moving phase that visits fewer nodes
2. **Better guarantees** — always produces connected communities (Louvain can produce disconnected ones)
3. **Refinement phase** — after finding communities, it randomly samples and refines the partition, avoiding poor local optima

Both maximize modularity, but Leiden's refinement step makes it more robust.

**Step 4 — Merge singletons** (`leiden.py:86-118`):

Single-node clusters are merged into the neighboring cluster with the strongest connection. The algorithm checks graph degree and finds the best target cluster via edge weights.

**Step 5 — Compute cluster quality** (`leiden.py:120-146`):

For each cluster, computes:
- `internal`: sum of edge weights within the cluster
- `external`: sum of edge weights crossing cluster boundaries
- `ratio`: internal / external — higher means tighter, more self-contained clusters

**Step 6 — Generate visualization** (`leiden.py:148-179`):

Converts igraph to NetworkX, uses SUMO node coordinates for positioning, color-codes by community, saves as PNG.

#### Output Format

Both algorithms produce the same JSON structure:

```json
{
  "clusters": {
    "0": ["node_intersection_A", "node_intersection_B", ...],
    "1": ["node_intersection_C", "node_intersection_D", ...]
  },
  "metrics": {
    "modularity": 0.85,
    "cluster_quality": {
      "0": {"internal": 120.5, "external": 30.2, "ratio": 3.99},
      "1": {"internal": 95.3, "external": 25.1, "ratio": 3.80}
    }
  }
}
```

For the OSM network (~1200 intersections), Leiden produces ~20-50 clusters depending on traffic conditions.

#### How It Feeds into the RL System

The cluster JSON defines the **hierarchy** for the meta-policy:
- Each cluster becomes one "subregion" processed by the TransformerEncoder
- Cluster-level traffic states (averaged 10-dim features per cluster) become the input to the meta-policy
- The number of clusters (`m`) determines the LSTM sequence length in SubGoalGenerator
- The adjacency between clusters (from the graph) defines the structure the GAT operates on

The granularity of clustering directly affects the RL system:
- **Too few clusters** → meta-policy has no meaningful hierarchy, acts like a flat controller
- **Too many clusters** → each cluster has too few intersections, meta-policy gets noisy signals

### Comparison: Manual vs Our Implementation

| Aspect | Manual (HiLight paper) | Our Implementation |
|--------|------------------------|-------------------|
| **Input** | Expert knowledge | SUMO network + 1000-step traffic simulation |
| **Adaptivity** | Static, one-time | Weights reflect actual traffic patterns |
| **Reproducibility** | Low — varies by expert | Deterministic given same network + seed |
| **Scalability** | Doesn't scale beyond small networks | Handles 1200+ intersections automatically |
| **Grounding** | Geographic/administrative intuition | Actual vehicle flow, waiting time, congestion |
| **Quality metric** | None | Modularity score + internal/external ratio |
| **Algorithm choice** | N/A | Louvain (fast, simple) or Leiden (robust, connected) |

### Suggested Improvements

The current implementation works but has clear areas for enhancement:

#### 1. Fix Louvain Dynamic Weights

The Louvain path doesn't actually use dynamic weights — `build_graph()` only stores `static_weight` (lane count) and `compute_weights()` is never called. Fix:

```python
# In LouvianService.build_graph(), add:
self.compute_weights()
# Then in to_weighted_undirected(), use self.edge_weights
```

#### 2. Adaptive Cluster Count

Currently the number of clusters is an output of the algorithm, not a parameter. For the RL system, a fixed number of clusters would simplify the TransformerEncoder and SubGoalGenerator. Options:
- **Resolution parameter** — both Louvain and Leiden accept a resolution/gamma parameter that controls cluster granularity (higher = more, smaller clusters)
- **Target K** — post-process by merging/splitting until reaching a target number (e.g., 20-30 clusters for the OSM network)
- **Learned clustering** — train a neural community detector end-to-end with the RL objective

#### 3. Time-Varying Clusters

Traffic patterns change hour-to-hour (morning rush vs night). The current approach computes weights over a fixed 1000-step window. Better:
- **Sliding window** — recompute clusters every N episodes during training
- **Periodic re-clustering** — run clustering at different times of day, store multiple cluster configs
- **Soft clustering** — instead of hard assignments, allow intersections to belong to multiple clusters with different weights

#### 4. Multi-Scale Hierarchy

A single level of clustering may not capture the right abstraction. Options:
- **Nested clusters** — run Leiden at multiple resolutions to build a cluster tree (coarse → fine)
- **Hierarchical attention** — the TransformerEncoder could process a 2-level hierarchy: coarse clusters → fine clusters → intersections
- **Adaptive depth** — learn when to use coarse vs fine-grained control

#### 5. Traffic-Aware Edge Weights

The current weight formula `α·vehicles + β·waiting + γ·congestion·vehicles` is hand-tuned. Improvements:
- **Learned weights** — treat α, β, γ as learnable parameters optimized alongside the RL agent
- **Time-dependent weights** — weight recent steps more heavily (exponential decay) to capture current conditions
- **Additional signals** — incorporate turn ratios, route choice, and OD (origin-destination) patterns

#### 6. Validation Framework

No automated evaluation of clustering quality exists. Add:
- **Baseline comparisons** — compare against geographic grid splitting, random partitioning, and spectral clustering
- **RL performance correlation** — measure whether better modularity scores lead to better RL convergence
- **Stability analysis** — run clustering multiple times with different simulation seeds, measure partition similarity (normalized mutual information)

#### 7. Direct RL-Driven Clustering

The ultimate improvement: eliminate the two-phase pipeline entirely. Instead of clustering first then training, learn the partition jointly with the RL agent:
- **Differentiable clustering** — use soft assignments (Gumbel-Softmax) so gradients flow through the partition
- **End-to-end objective** — the clustering is optimized to maximize traffic throughput, not modularity
- **Dynamic repartitioning** — the agent learns to reorganize clusters as traffic conditions change
