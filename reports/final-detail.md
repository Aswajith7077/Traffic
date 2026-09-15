# Research Foundation Report

## Hierarchical, Region-Aware Multi-Agent Reinforcement Learning for Adaptive Traffic Signal Control

**Repository:** `Traffic`

**Assessment type:** Current implementation and research-baseline assessment

**Intended audience:** Research foundation and board members

**Assessment date:** 2026-09-06

**Baseline status:** The repository contains a substantial research prototype and an executable SUMO-based pipeline. It should be treated as a promising engineering and experimentation foundation, not yet as a validated production controller or as a faithful, fully operational reproduction of the HiLight method.

---

## 1. Executive Summary

This project investigates hierarchical reinforcement learning for adaptive traffic signal control. Its central design is to divide a road network into regions, use a **meta-policy** to reason over regional traffic conditions, and use a **sub-policy** to select signal actions at individual intersections. The implementation is built around SUMO and TraCI and includes:

- A region-splitting module with DBSCAN, Leiden, Louvain, and four hybrid partitioning pipelines.
- A SUMO/TraCI environment with intersection observations, traffic-light phase control, minimum-green and yellow-transition handling, and multi-objective reward terms.
- A Transformer-based regional encoder and LSTM subgoal generator.
- A local MLP encoder, graph-attention layer, and shared Actor-Critic action model.
- Episode-based training, checkpoint creation, evaluation scripts, a GUI runner, and an end-to-end pipeline wrapper.
- Five configured scenarios: Manhattan, Cologne8, Ingolstadt21, Arterial4x4, and Grid4x4.

The project is strongest as a **modular research platform**. The region partitioning component is sufficiently broad to support comparative research on topology-based, spatial, and hybrid clustering. The SUMO integration provides a realistic simulation loop and exposes useful traffic features. The codebase also contains important foundations for fairness, emergency priority, pedestrian conflict handling, and emissions-related objectives.

However, the current implementation has several limitations that materially affect the interpretation of results:

1. The hierarchical control loop is only partially connected. The meta-policy generates a subgoal, but that subgoal is not passed into the Actor-Critic input during either training or evaluation.
2. The Actor-Critic has no optimizer step. The current training loop creates optimizers for the Transformer, subgoal generator, local encoder, and GAT, but not for the Actor-Critic itself.
3. Replay-buffer states are detached before storage, preventing the intended end-to-end gradient path through the local encoder and GAT.
4. The action head always emits seven actions even though intersections may have different valid phase sets.
5. Several traffic features and evaluation metrics do not yet have reliable semantics. Examples include cluster-level flow, cluster queue length, phase entropy, and the evaluation queue metric.
6. Current results are not controlled enough to establish superiority over the actuated SUMO baseline. Evaluation horizons, completed-vehicle counts, seeds, checkpoints, and demand conditions vary.
7. The current environment does not yet implement the full future research agenda: fairness for every vehicle as a formal guarantee, automated control-aware clustering, explicit emergency-vehicle communication/priority logic, or accident-zone detection and adaptive optimization.

The appropriate board-level conclusion is therefore:

> The project has a credible technical foundation for a research program in scalable, hierarchical traffic signal control. The immediate priority is not to claim final performance gains, but to repair the learning pathway, establish reproducible baselines, validate the partitioning objective, and then add fairness, emergency-vehicle, and incident-aware capabilities through measurable research phases.

---

## 2. Research Problem and Intended Contribution

### 2.1 Problem statement

Large urban traffic networks create a coordination problem. A centralized controller must process a very large state and action space, while purely independent local controllers can make decisions that improve one intersection and worsen congestion elsewhere. The project addresses this tension with a hierarchy:

- The **regional level** observes aggregate conditions over groups of intersections.
- The **local level** observes individual intersection conditions and neighboring intersection representations.
- Regional information is intended to guide local action selection through subgoals.

The target outcomes are reduced congestion, lower average trip time, lower delay and waiting time, improved network throughput, and more equitable service across vehicles. The repository documentation additionally identifies emergency-vehicle priority and emissions reduction as intended objectives.

### 2.2 Claimed conceptual basis

The project uses the HiLight hierarchical reinforcement-learning paper as a conceptual reference. The documented design is consistent with the following ideas from that family of approaches:

- Transformer-based encoding of regional traffic state.
- LSTM-based generation of temporal subgoals.
- Graph-aware local representations for neighboring intersections.
- Shared local policy parameters to reduce policy size and improve transferability.
- A meta-policy and sub-policy relationship intended to behave adversarially: the meta-policy presents challenging goals and the sub-policy attempts to satisfy or exceed them.

The repository should not be described as a complete reproduction of the paper. The current code includes structural equivalents of several components, but the action-conditioning and optimization pathways do not yet realize the full intended hierarchical/adversarial interaction.

### 2.3 Intended research contribution

The research foundation proposed by this codebase can be framed as four linked contributions:

1. **Scalable hierarchy:** Use learned regional abstractions to reduce the effective complexity of large traffic networks.
2. **Partitioning comparison:** Compare spatial density clustering, graph community detection, and their combinations rather than assuming one fixed partition is optimal.
3. **Multi-objective control:** Extend traffic-signal optimization beyond aggregate delay to include vehicle-level fairness, emergency response, and incident robustness.
4. **Adaptive network intelligence:** Move from static regions and nominal traffic assumptions toward automatically updated regions, detected accident zones, and context-aware control.

Only the first two contributions are substantially represented in the current implementation. The third is partially represented in reward code. The fourth remains a proposed research direction.

---

## 3. Repository Scope and System Boundary

The repository is organized as two independent but connected modules.

### 3.1 Region-splitting module

Location: `region-splitting/`

Purpose:

- Load SUMO network topology and coordinates.
- Build graph or spatial representations.
- Generate candidate regions using multiple clustering families.
- Compute partition metrics and visualization artifacts.
- Write cluster JSON files consumed by the RL module.

Primary entry points:

- `region-splitting/main.py`
- `region-splitting/generate_clusters.py`
- `region-splitting/generate_dbscan_clusters.py`

### 3.2 Adversarial/RL module

Location: `advesarial/`

The directory name is intentionally misspelled in the repository and should be retained in paths unless a deliberate migration is planned.

Purpose:

- Load a selected scenario and cluster partition.
- Start SUMO through TraCI.
- Observe intersection and cluster-level traffic state.
- Train hierarchical neural components.
- Evaluate checkpoints using traffic metrics.

Primary entry points:

- `advesarial/src/sample.py` for training.
- `advesarial/src/evaluate.py` for evaluation.
- `advesarial/run_gui.py` for visualization/playback.

### 3.3 Orchestration and scenarios

The root `pipeline.py` coordinates baseline simulation, clustering, cluster-file transfer, training, and evaluation. Scenario data is stored under `scenarios/<scenario>/`.

Configured scenarios:

| Scenario | Role in current repository |
|---|---|
| Manhattan | Main large urban reference and default scenario |
| Cologne8 | Smaller benchmark used in recent method comparisons |
| Ingolstadt21 | Additional urban benchmark |
| Arterial4x4 | Synthetic/structured arterial benchmark with route variants |
| Grid4x4 | Supported for partitioning, but limited for RL where traffic-light coverage is absent or unsuitable |

The code supports more than Manhattan, but the research evidence and historical development emphasis are uneven. Manhattan remains the most important reference case, while Cologne8 currently provides the clearest method-comparison artifacts.

---

## 4. End-to-End Data and Execution Flow

### 4.1 Standard workflow

The intended sequence is:

1. Select a scenario and clustering method.
2. Run region splitting against the scenario network.
3. Copy the generated cluster JSON into `advesarial/clusters/<method>/`.
4. Set `TRAFFIC_SCENARIO` and `CLUSTER_METHOD`.
5. Train from the `advesarial/` module root using `src/sample.py`.
6. Evaluate a checkpoint using `src/evaluate.py`.

The root pipeline automates these phases:

```bash
python pipeline.py --scenario cologne8 --cluster-method dbscan_leiden
```

The implementation of this orchestration is in `pipeline.py:90-208` and the command-line options are defined at `pipeline.py:211-301`.

### 4.2 Cluster configuration

`advesarial/src/config.py` reads:

```text
clusters/<method>/<scenario>_clusters.json
```

The default values are:

- `TRAFFIC_SCENARIO=manhattan`
- `CLUSTER_METHOD=dbscan`

If the requested cluster file is absent, the configuration falls back to DBSCAN and then Leiden. This makes the system easier to run, but it also creates a scientific risk: a missing requested partition can silently change the experiment. Research runs should fail loudly when an explicitly requested cluster method is unavailable.

The cluster JSON format is approximately:

```json
{
  "clusters": {
    "0": ["traffic_light_id_1", "traffic_light_id_2"],
    "1": ["traffic_light_id_3"]
  },
  "metrics": {
    "...": "..."
  }
}
```

At runtime, cluster nodes are filtered against traffic-light IDs discovered from SUMO. There is no strong validation that all live intersections belong to exactly one cluster, that there are no duplicate memberships, or that the partition matches the checkpoint architecture.

### 4.3 SUMO and TraCI

`advesarial/src/services/traci.py` provides the simulation boundary. It:

- Validates `SUMO_HOME`.
- Starts `sumo` or `sumo-gui`.
- Loads traffic-light IDs and road connectivity.
- Reads lane, edge, waiting, speed, occupancy, queue, pressure, and phase information.
- Applies discrete signal phase actions.
- Handles yellow transitions and minimum-green timing.
- Resets simulation state between episodes.

The main training scenario path is assembled as:

```text
../scenarios/<scenario>/<scenario>.sumocfg
```

This path is relative to the process working directory. The required convention is to run training from `advesarial/`, not from `advesarial/src/`.

The dependence on the caller's working directory is a reproducibility and deployment concern. It affects training, evaluation, cluster generation, and package entry points differently.

---

## 5. Region-Splitting Implementation

The repository intentionally explores multiple definitions of a useful traffic-control region. This is a central research strength because geographic compactness, road connectivity, and dynamic congestion structure are not the same objective.

### 5.1 Common service interface

`region-splitting/services/base.py` defines the shared conceptual interface:

```text
build_graph() -> get_clusters() -> generate_visualization()
```

`services/registry.py` maps method names to implementations. The available method names are:

- `leiden`
- `louvian`
- `dbscan`
- `dbscan_louvian`
- `dbscan_leiden`
- `louvian_dbscan`
- `leiden_dbscan`

The spelling `louvian` is used throughout the current project, although the standard algorithm name is Louvain.

### 5.2 Leiden

Implementation: `region-splitting/services/leiden.py`

The Leiden path:

1. Builds an `igraph` graph from non-special SUMO edges.
2. Computes traffic-derived edge weights from a SUMO simulation.
3. Uses `leidenalg.ModularityVertexPartition`.
4. Merges selected low-degree singleton communities into neighboring communities.
5. Reports modularity and cluster-level internal/external edge-weight information.

Dynamic weights combine traffic indicators such as vehicle count, waiting time, speed reduction, and congestion. This makes Leiden the most explicitly traffic-aware of the standalone community-detection paths.

Limitations:

- The partition depends on a simulation window and demand state.
- Randomness is not consistently controlled.
- Modularity is not a direct measure of signal coordination quality.
- Very fine partitions have appeared in historical artifacts, including approximately 1,974 Leiden regions for Manhattan, which is not suitable for the intended regional meta-policy without further aggregation.

### 5.3 Louvain

Implementation: `region-splitting/services/louvian.py`

The Louvain path constructs a NetworkX graph and performs modularity optimization, followed by merging of small communities.

There is a material implementation issue: graph construction stores `static_weight`, while the conversion path searches for `lane_weight` and falls back to a default weight. In addition, the dynamic `compute_weights()` facility is not called by the standard path. Consequently, current Louvain results should be treated as mostly topology-based rather than reliably traffic-weighted.

This is important for research interpretation. A comparison described as “traffic-weighted Louvain versus DBSCAN” would not be accurate unless the weight-key and invocation issues are repaired and the run is regenerated.

### 5.4 DBSCAN

Implementation: `region-splitting/services/dbscan.py`

DBSCAN clusters network points using spatial coordinates. It includes:

- `eps` and `min_samples` configuration.
- Noise labeling.
- Optional reassignment of noise to the nearest cluster centroid.
- Silhouette, Davies-Bouldin, Calinski-Harabasz, size, density, and noise metrics.
- Parameter sensitivity analysis.
- K-distance and elbow-support utilities.
- Stability and grid-search diagnostics.
- Synthetic tests in `region-splitting/tests/test_dbscan.py`.

The production generator auto-tunes parameters using heuristic target ranges:

- A desired average cluster size range.
- A maximum noise fraction.
- A candidate search over `eps` values.
- A `min_samples` value scaled to network size.

DBSCAN is therefore the current production default and is reported in the repository as producing coarse, spatially compact regions. For the checked-in Manhattan artifact, the DBSCAN partition contains approximately 72 clusters and no remaining noise after reassignment.

The principal limitation is that DBSCAN is spatial, not road-topological. Two intersections can be geographically close while being poorly connected for signal coordination. Conversely, a long corridor can be topologically important while exceeding the spatial radius. The present auto-tuning objective optimizes geometric clustering statistics rather than downstream traffic-control performance.

### 5.5 Hybrid methods

Implementation: `region-splitting/services/hybrid.py`

The four hybrid methods use two processing directions.

#### DBSCAN first, community detection second

For `dbscan_louvian` and `dbscan_leiden`:

1. DBSCAN creates spatial cells.
2. Cells are contracted into a supernode graph.
3. Louvain or Leiden partitions the supernode graph.
4. Supernode communities are expanded back to network nodes.

#### Community detection first, DBSCAN second

For `louvian_dbscan` and `leiden_dbscan`:

1. A graph community partition is generated.
2. DBSCAN is run independently within each community.
3. The subregions are combined into the final partition.

The module reports coverage, cluster sizes, silhouette, Davies-Bouldin, and Calinski-Harabasz scores. These metrics are useful diagnostics but do not establish that the partition is good for traffic-signal control.

Potential coverage risks exist during graph contraction and community expansion, especially for isolated clusters or nodes not represented in the intermediate edge graph. The final partition must be validated for complete, non-overlapping traffic-light coverage before every training run.

### 5.6 Current clustering research position

The codebase already supports a meaningful clustering research question:

> Which region construction objective produces the best scalable control policy under matched network, demand, training budget, and evaluation conditions?

The answer cannot be inferred from silhouette or modularity alone. The required final objective should include:

- Traffic-light coverage.
- Region connectivity and boundary count.
- Spillback and queue propagation across boundaries.
- Phase coordination value.
- Partition stability across seeds and demand conditions.
- Meta-policy input size and computational cost.
- Downstream travel time, delay, fairness, emergency response, and robustness.

---

## 6. Hierarchical RL Implementation

### 6.1 Regional/meta-policy path

The regional path is implemented in:

- `advesarial/src/models/encoder/transformer.py`
- `advesarial/src/models/lstm.py`
- `advesarial/src/sample.py`

Each cluster is represented by a 10-dimensional aggregate state. The Transformer:

1. Projects cluster features into a model dimension.
2. Prepends a learned global token.
3. Adds sinusoidal positional encoding.
4. Applies Transformer encoder layers.
5. Returns a global representation and per-cluster embeddings.

The training setup uses a 128-dimensional model, eight attention heads, and six Transformer layers. The LSTM subgoal generator processes cluster embeddings and combines the resulting representation with the global embedding. The output is intended to encode regional waiting and queue goals.

The positional encoding has a default capacity of approximately 200 positions, including the global token. A partition with more than approximately 199 clusters can therefore exceed the configured positional range. This is especially relevant because historical Leiden partitions can be extremely fine.

### 6.2 Local/sub-policy path

The local path is implemented using:

- `LocalEncoder`: 10 input features to a 64-dimensional representation.
- `GATLayer`: one graph-attention aggregation layer over neighboring intersections.
- A network-level mean feature concatenated with each node representation.
- `ActorCritic`: a shared model producing action probabilities and a scalar value estimate.

The effective local input is approximately 128 dimensions: 64 dimensions from the local graph representation plus 64 dimensions from the global mean representation.

The GAT implementation is loop-based and likely to be costly for larger networks. It is also a simple single-layer attention mechanism rather than a fully optimized multi-head graph network.

### 6.3 Action model

The Actor-Critic action head emits seven action probabilities. TraCI maps action indices to valid signal phases using modulo behavior.

This is not equivalent to a per-intersection valid-action policy. Different intersections can have different phase counts and phase semantics. Modulo mapping can make multiple action indices map to the same phase or can cause the policy to assign probability mass to semantically invalid choices.

A research-grade implementation should provide:

- A per-intersection action mask.
- Explicit phase identity and transition constraints.
- A policy head that understands each intersection's valid action set.
- Consistent handling of yellow and minimum-green intervals.
- A logged action-to-phase mapping for every experiment.

### 6.4 Hierarchy-coupling gap

The most important conceptual gap is in `sample.py:118-133`. The training loop computes a meta-policy subgoal, but the Actor-Critic is called with the local state without that subgoal. Evaluation similarly recomputes and discards the global subgoal.

The current execution is therefore structurally hierarchical but behaviorally closer to:

```text
regional model: trained auxiliary objective
local policy: selects actions from local/graph state
```

It is not yet:

```text
regional goal -> goal-conditioned local policy -> traffic action
```

This distinction should be explicit in all board and research materials.

### 6.5 Optimization gap

Four optimizers are created for:

- Transformer.
- Subgoal generator.
- Local encoder.
- GAT.

There is no Actor-Critic optimizer. Consequently, the policy and value network parameters are not stepped by the training loop. Replay states are also detached before storage, so gradients cannot propagate from later policy losses through the encoder and GAT into the representation pipeline.

This makes the current training results unsuitable as evidence of a successfully optimized hierarchical policy. The issue is repairable, but all performance comparisons should be regenerated after the repair.

### 6.6 Losses and adversarial interpretation

The active losses include:

- A meta loss aligning subgoals with global waiting and queue states.
- A TD-style value loss.
- A policy-gradient actor loss.
- A subgoal-alignment loss.

The implementation does not contain a separate adversarial game with clearly separated maximization/minimization updates, a goal-challenging meta objective, or a formal constrained adversarial curriculum. The term “adversarial” is currently best interpreted as the intended meta/sub-policy research design rather than as a fully realized adversarial optimization mechanism.

---

## 7. Environment, Observations, and Reward

### 7.1 Intersection observation

`TraciService` constructs a 10-dimensional observation per traffic light. The intended features are:

1. Vehicle count.
2. Halted/queued vehicle count.
3. Occupancy.
4. Flow.
5. Stop count.
6. Waiting time.
7. Average speed.
8. Pressure.
9. Congestion ratio.
10. Delay.

The semantic implementation of some features needs verification. In particular, the flow value is currently another vehicle-count-like measurement rather than a robust in/out flow estimate.

### 7.2 Cluster observation

Cluster states aggregate conditions across incident edges and are also represented with ten values. Several fields are currently inconsistent:

- A field labeled as total queue length accumulates occupancy-like values.
- Incoming and outgoing flow are incremented from the same vehicle count, so internal cluster flow can collapse to zero.
- Phase entropy depends on phase histories that are not updated in the active path.

Before any scientific analysis uses these features, each feature needs a unit definition, source TraCI call, aggregation rule, and validation test.

### 7.3 Reward composition

The environment reward is intended to combine:

- Efficiency.
- Fairness.
- Emergency delay.
- Emissions-related proxy terms based on stops.
- Pedestrian conflict penalties in relevant logic.

The documented weighting is approximately:

0.4 efficiency + 0.2 fairness + 0.3 emergency + 0.1 emissions
```

The reward also applies normalization using moving statistics.

This is a good foundation for a multi-objective research program, but the weights are not yet justified through sensitivity analysis, preference elicitation, safety constraints, or Pareto-front experiments.

### 7.4 Fairness status

Fairness is present as a reward concept, including Jain-style utility logic and waiting-related terms. It should not yet be described as “fairness for every vehicle” in the strong research sense because:

- There is no persistent per-vehicle service history.
- There is no formal maximum-wait constraint.
- There is no worst-served percentile objective.
- There is no protected-class or route-class fairness definition.
- Vehicle fairness is not separately reported in evaluation.
- A utility variable is overwritten across intersections, making the implemented aggregation questionable.

The current implementation provides a starting point for fairness-aware reward shaping, not a validated fairness guarantee.

---

## 8. Training, Checkpointing, and Evaluation

### 8.1 Training loop

`advesarial/src/sample.py` runs episode-based SUMO training. Configurable environment variables include episode count, episode length, save interval, scenario, route, and cluster method.

The training loop:

1. Starts the SUMO-backed environment.
2. Resets the environment for each episode.
3. Computes regional and local representations.
4. Samples signal actions.
5. Advances SUMO.
6. Stores transitions in replay memory.
7. Samples mini-batches after a warm-up threshold.
8. Updates selected optimizers.
9. Saves periodic checkpoints.

Checkpoints contain model weights, optimizer states, episode information, timestamps, and some environment parameters. Model directories use timestamps and random suffixes.

### 8.2 Training instability

Historical logs contain episodes with very large Actor-Critic losses, negative rewards, and large meta/subgoal losses. Later runs show lower loss magnitudes in some sessions, but the available metrics do not establish stable convergence.

The most likely contributors are:

- Missing Actor-Critic optimizer.
- Detached replay representations.
- Unconnected subgoal pathway.
- One-step TD targets and unbounded scale differences.
- No entropy regularization or GAE in the active loop.
- Inconsistent observation/reward normalization.
- Stochastic training without recorded seeds.

### 8.3 Evaluation

The evaluation code creates a fresh environment and reconstructs the model. It can inspect checkpoint dimensions and reports:

- Completed vehicles.
- Average travel time.
- Delay relative to estimated free-flow travel time.
- Waiting time.
- Queue statistics.
- Peak queue.

The policy uses greedy `argmax` actions. This is reasonable for deterministic evaluation, but the evaluation protocol must be fixed across all methods and seeds.

There is a Python 3 syntax error in `evaluate.py` in the exception clause around line 221, and the package entry point targets `evaluate:main` even though the file does not expose the expected `main()` function. Evaluation must be repaired and tested before it is used as an automated research gate.

Evaluation averages can be misleading because they are computed over vehicles completed within the evaluation horizon. A controller that completes fewer vehicles can appear favorable or unfavorable depending on which vehicles finish. Every evaluation should report:

- Total demand.
- Departed vehicles.
- Completed vehicles.
- Vehicles still in network.
- Completion rate.
- Mean and percentile travel time.
- Mean and percentile delay.
- Mean and maximum waiting time.
- Throughput.
- Queue and spillback metrics.

---

## 9. Current Empirical Evidence

### 9.1 Available baseline evidence

The repository contains actuated SUMO baseline records for several scenarios:

| Scenario | Completed vehicles | Average travel time | Average waiting time |
|---|---:|---:|---:|
| Arterial4x4 | 1,150 | 830.22 s | 589.41 s |
| Cologne8 | 1,995 | 114.39 s | 29.27 s |
| Ingolstadt21 | 3,997 | 295.31 s | 103.01 s |
| Manhattan | 1,800 | 223.41 s | 103.23 s |

These values are recorded in `advesarial/metrics.txt` and corroborating scenario `stats.xml` files. They establish a useful starting baseline, but not a complete benchmark suite. There are no consistently recorded fixed-time, Webster, max-pressure, or other classical baselines.

### 9.2 Reported RL and clustering results

The repository contains historical Manhattan evaluations and later cross-method evaluations. Recent Cologne8 records include examples such as:

| Partition | Completed vehicles | Average travel time | Average delay |
|---|---:|---:|---:|
| DBSCAN | 132 | 164.26 s | 117.67 s |
| DBSCAN to Louvain | 263 | 109.46 s | 60.20 s |
| DBSCAN to Leiden | 265 | 111.40 s | 61.95 s |
| Louvain to DBSCAN | 359 | 191.90 s | 144.30 s |
| Leiden to DBSCAN | 263 | 109.46 s | 60.20 s |

These results are useful exploratory evidence that partition choice changes behavior. They are not yet fair comparative experiments because horizons, checkpoints, training durations, route/demand conditions, seeds, and completed-vehicle populations are not controlled consistently.

### 9.3 What the current evidence supports

The evidence supports these limited conclusions:

- The pipeline can generate and consume multiple cluster types.
- Different cluster partitions can materially change the observed control behavior.
- The project can run multi-scenario SUMO experiments.
- A multi-objective reward and hierarchical model scaffold exist.
- The current code and logs expose meaningful research questions.

The evidence does not yet support these stronger conclusions:

- The RL controller consistently outperforms actuated control.
- The meta-policy improves the local policy.
- DBSCAN is globally optimal.
- Hybrid methods are superior across networks.
- Fairness is guaranteed for every vehicle.
- Emergency vehicles receive reliable priority.
- Accident zones are detected and optimized around.
- The repository reproduces the published HiLight results.

---

## 10. Limitations and Technical Risks

### 10.1 Critical learning-path risks

1. **Actor-Critic not optimized:** no policy/value optimizer is stepped.
2. **Detached state path:** replay storage breaks gradient propagation through local encoders.
3. **Subgoal not used for action:** regional guidance is auxiliary rather than control-causal.
4. **Fixed seven-action output:** does not match variable intersection phase spaces.
5. **Unclear adversarial optimization:** no explicit meta-versus-sub-policy game is present.
6. **Replay semantics:** stored state representations can come from older model parameters, complicating off-policy updates.

### 10.2 Observation and metric risks

1. Cluster queue and flow fields are not consistently defined.
2. Phase entropy is not reliable because histories are not updated.
3. Evaluation queue currently counts vehicles on controlled lanes rather than only halted vehicles.
4. Free-flow travel-time calculation assumes a lane naming convention that may not hold everywhere.
5. Metrics are appended to text files without a stable run schema.
6. Results frequently omit seed, commit, route, demand, checkpoint, and simulator version.

### 10.3 Clustering risks

1. DBSCAN is spatial and not road-connectivity-aware.
2. Louvain weighting has a key mismatch and is not reliably dynamic.
3. Leiden can produce excessively fine partitions.
4. Hybrid expansion may not guarantee complete coverage.
5. Auto-tuning optimizes geometric diagnostics rather than downstream control value.
6. Region count changes the neural architecture and can invalidate checkpoints.

### 10.4 Software and reproducibility risks

1. Relative paths depend on the current working directory.
2. Importing `sample.py` performs substantial initialization and can start SUMO before `main()`.
3. `traffic-eval` points to a missing `main()` function.
4. `evaluate.py` contains invalid Python 3 exception syntax.
5. `train.py` is a broken legacy path and should not be treated as supported.
6. Duplicate encoders, unused schemas, and stubs create ambiguity about the active architecture.
7. Random seeds are not consistently configured.
8. Model checkpoints are ignored from version control, making provenance dependent on external storage.
9. There is only one meaningful automated test file, focused on synthetic DBSCAN behavior.

---

## 11. Proposed Research Expansion

The board's proposed direction is technically compatible with the repository, but should be implemented as explicit research phases rather than as undocumented additions to the current reward.

### 11.1 Fairness for every vehicle

The project should define fairness at the vehicle service level, not only as a global aggregate reward.

Recommended definitions:

- Per-vehicle waiting time.
- Per-vehicle delay relative to free-flow travel time.
- Maximum waiting time.
- 90th, 95th, and 99th percentile delay.
- Jain's fairness index over normalized service utility.
- Route-level and origin-destination-level fairness.
- Fairness for vehicles that cross region boundaries.

Recommended control design:

- Maintain vehicle-level or class-level service histories.
- Add a bounded fairness penalty or constrained optimization term.
- Use lexicographic safety rules for extreme waiting, rather than relying only on a weighted sum.
- Prevent fairness from being achieved by reducing throughput for all vehicles.
- Report the efficiency/fairness Pareto frontier.

Required experiments:

- Aggregate reward only.
- Reward plus Jain fairness.
- Reward plus tail-wait constraint.
- Fairness-aware controller under unequal route demand.
- Fairness under changing demand and region partitions.

### 11.2 Automated, control-aware clustering

The current clustering methods are valuable candidates, but the future system should choose or update partitions based on control usefulness.

Potential features:

- Road topology and directed connectivity.
- Physical distance.
- Queue spillback correlation.
- Travel-time correlation.
- Phase compatibility.
- Shared route demand.
- Boundary-crossing flow.
- Emergency corridor relationships.
- Incident propagation risk.

Potential automated approaches:

1. **Control-aware parameter tuning:** choose DBSCAN parameters using downstream delay, spillback, fairness, and computational cost.
2. **Graph neural partitioning:** learn region assignments with connectivity and capacity constraints.
3. **Online adaptive clustering:** update regions when traffic correlations change while limiting assignment churn.
4. **Multi-objective partition selection:** maintain a Pareto set over compactness, connectivity, stability, and control performance.
5. **Hierarchical partitioning:** use coarse stable regions for the meta-policy and finer transient subregions for incident response.

Every clustering output should include a manifest with:

- Scenario and network hash.
- Demand/route file.
- Method and parameters.
- Random seed.
- Node coverage.
- Duplicate membership count.
- Connectivity statistics.
- Cluster size statistics.
- Downstream evaluation identifier.

### 11.3 Emergency vehicles (EMV)

Emergency-vehicle handling should be represented as a complete sensing, prioritization, and evaluation path.

Required components:

- Emergency vehicle identity and class in SUMO demand.
- Current lane, route, destination, and estimated arrival path.
- Detection radius and time-to-intersection features.
- Emergency phase request or priority state.
- Conflict-resolution rules when multiple emergency vehicles compete.
- Recovery logic preventing persistent priority-induced starvation.
- Explicit emergency travel-time and delay metrics.

Possible policy design:

- Add an emergency context vector to regional and local observations.
- Add emergency corridor objectives to the meta-policy.
- Use constrained phase selection for imminent emergency arrivals.
- Reserve or dynamically prioritize a corridor across region boundaries.
- Penalize emergency delay strongly, but bound the cost imposed on general traffic.

Required experiments:

- No emergency vehicles.
- One emergency vehicle.
- Multiple emergency vehicles on intersecting routes.
- Emergency demand under high congestion.
- Emergency priority with and without fairness constraints.

### 11.4 Accident-zone detection and incident-aware optimization

Accident handling is not currently implemented and should be treated as a distinct research module.

Suggested incident signals:

- Sudden speed reduction.
- Abnormal occupancy increase.
- Stationary vehicles.
- Lane blockage or capacity reduction.
- Queue growth inconsistent with upstream demand.
- Abrupt route completion or flow changes.
- External incident labels in SUMO.

An incident detector should produce:

- Incident probability.
- Affected edge/lane/intersection set.
- Incident severity.
- Estimated duration.
- Confidence and uncertainty.

The controller should then:

- Recompute or locally refine regions.
- Protect upstream intersections from spillback.
- Reroute or meter traffic where supported.
- Prioritize emergency access.
- Avoid using a stale nominal policy in the affected zone.
- Preserve fairness for vehicles diverted around the incident.

Evaluation should compare:

- Normal traffic without incidents.
- Incident detection delay.
- Incident-zone queue growth.
- Network recovery time.
- Emergency response time.
- Total travel-time increase relative to no-incident conditions.
- Fairness impact on unaffected and diverted vehicles.

### 11.5 Multi-objective and constrained formulation

The current weighted reward is a practical prototype, but the future research should compare three formulations:

1. Weighted-sum reward.
2. Constrained RL with explicit maximum-delay and emergency constraints.
3. Pareto or preference-conditioned policy learning.

This will help the foundation distinguish a controller that merely optimizes a selected scalar score from one that reliably respects public-interest constraints.

---

## 12. Recommended Stabilization and Research Roadmap

### Phase 0: Reproducibility and execution hygiene

Priority: immediate

- Repair Python 3 syntax in evaluation and comparison utilities.
- Add real `main()` functions or correct package entry points.
- Move all path resolution to repository-root-relative paths.
- Make missing requested cluster files fatal in research mode.
- Record git revision, SUMO version, Python/dependency versions, route, seed, cluster method, and checkpoint.
- Add deterministic seed configuration for Python, NumPy, PyTorch, SUMO, and clustering algorithms.
- Replace append-only free-form metrics with JSON Lines or CSV plus run metadata.

Acceptance criteria:

- A clean environment can run baseline, cluster, train, and eval from documented commands.
- The same configuration and seed produce a traceable, reproducible run.
- An evaluation run fails clearly when artifacts are missing or incompatible.

### Phase 1: Correct the learning pathway

Priority: immediate

- Add and step an Actor-Critic optimizer.
- Decide whether training is on-policy or off-policy and align replay accordingly.
- Preserve or recompute differentiable representations correctly.
- Pass meta-policy subgoals into the local Actor-Critic.
- Add explicit dimensions and checkpoint validation.
- Implement per-intersection action masks.
- Add entropy regularization, stable advantage estimation, and terminal handling.
- Log gradient norms and parameter-update magnitudes.

Acceptance criteria:

- Actor-Critic parameter norms change during training.
- Local encoder and GAT receive nonzero verified gradients.
- Subgoal perturbation changes local policy outputs in a controlled test.
- Invalid signal phases receive zero probability.

### Phase 2: Metric and environment validation

Priority: immediate

- Define every observation field and unit.
- Correct cluster queue and flow aggregation.
- Update phase histories and validate entropy.
- Correct evaluation queue semantics.
- Validate free-flow travel-time estimation against all scenario lane IDs.
- Add unit tests for reward components and edge cases.

Acceptance criteria:

- Synthetic TraCI fixtures produce expected observations and rewards.
- Every evaluation metric has a documented numerator, denominator, unit, and population.

### Phase 3: Controlled baseline study

Priority: high

Compare, under identical scenario, demand, horizon, seeds, and completed-vehicle accounting:

- Fixed-time control.
- Actuated SUMO control.
- Max-pressure or another classical pressure controller.
- Independent local RL.
- Hierarchical RL without meta guidance.
- Hierarchical RL with active meta guidance.

Report means, standard deviations, confidence intervals, and effect sizes over multiple seeds. Include throughput and completion rate, not only averages among completed vehicles.

### Phase 4: Partition study

Priority: high

Run all seven partition methods with matched model and training budgets. Measure:

- Partition quality.
- Coverage and connectivity.
- Training cost.
- Inference cost.
- Meta-policy size.
- Travel time and delay.
- Fairness.
- Emergency response.
- Robustness to demand shift.

The central result should be a downstream control comparison, not a claim that one clustering metric is universally optimal.

### Phase 5: Fairness and emergency vehicles

Priority: high

- Add vehicle-level service tracking.
- Add EMV demand and route scenarios.
- Implement corridor-level emergency priority.
- Compare reward shaping with constrained control.
- Report tail service outcomes and emergency-specific outcomes.

### Phase 6: Accident-zone intelligence

Priority: medium/high

- Generate or import incident scenarios.
- Implement incident detection and confidence.
- Add adaptive local partitioning and incident-aware subgoals.
- Evaluate detection, response, recovery, safety, and fairness.

### Phase 7: External validity and deployment research

Priority: later

- Test additional networks and demand regimes.
- Evaluate computational scaling beyond current scenario sizes.
- Study sim-to-real transfer assumptions.
- Add safety shields and human-operational constraints.
- Establish governance for emergency and fairness tradeoffs.

---

## 13. Recommended Research Claims at This Stage

The following claims are supported by the current repository:

- A modular SUMO-based hierarchical traffic-control research platform has been implemented.
- Multiple automatic and hybrid network partitioning methods are available.
- The platform can generate, transfer, train on, and evaluate cluster-based traffic-control configurations.
- The architecture contains Transformer, LSTM, graph-attention, and Actor-Critic components corresponding to the intended hierarchical design.
- Multi-objective reward scaffolding includes efficiency, fairness-related, emergency-related, and emissions-proxy terms.
- Preliminary logs show that partition choice and scenario choice materially affect behavior.

The following claims should be deferred until the roadmap is completed:

- “The controller reduces congestion across scenarios.”
- “The meta-policy improves the sub-policy.”
- “The approach is adversarial multi-agent RL” in the strict optimization sense.
- “The clustering approach is optimal.”
- “Every vehicle receives fair treatment.”
- “Emergency vehicles are reliably prioritized.”
- “Accident zones are detected and optimized in real time.”
- “The implementation reproduces HiLight results.”

The suitable present-tense framing is **research prototype**, **baseline platform**, or **experimental foundation**.

---

## 14. Board-Level Assessment

### Strengths

- Clear and relevant societal problem.
- Ambitious but coherent hierarchical architecture.
- Realistic SUMO/TraCI simulation integration.
- Broad region-splitting research surface.
- Existing multi-scenario assets and generated cluster artifacts.
- Checkpointing, evaluation, GUI, and orchestration foundations.
- Natural path to fairness, EMV, and incident-aware extensions.

### Current concerns

- Core policy learning is not yet technically valid because the Actor-Critic is not optimized.
- The meta-policy is not causally connected to local actions.
- Current metrics are not sufficient for scientific performance claims.
- Scenario and cluster comparisons are not yet controlled.
- Several software paths are fragile or broken.
- The future fairness, EMV, and accident objectives are mainly planned rather than implemented.

### Funding/research recommendation

The project merits continued research investment as a foundation for a staged program. Funding should be tied to technical gates rather than to immediate claims of traffic improvement:

1. Make the current pipeline executable and reproducible.
2. Repair and validate the hierarchical learning pathway.
3. Establish controlled baselines and multi-seed evidence.
4. Demonstrate partition selection based on downstream control value.
5. Add and evaluate vehicle-level fairness.
6. Add emergency vehicles and incident-aware adaptation.
7. Publish only claims supported by controlled experiments and clear statistical evidence.

The project is well positioned to evolve from a Manhattan-centered prototype into a broader adaptive traffic-intelligence framework. The key requirement is disciplined separation between what the current code already implements, what is currently experimental, and what is part of the proposed research agenda.

---

## Appendix A. Key Files Reviewed

### Core orchestration and documentation

- `AGENTS.md`
- `README.md`
- `summary.md`
- `details.md`
- `pipeline.py`
- `pyproject.toml`
- `metrics.txt`
- `advesarial/metrics.txt`
- `plans/actor-critic-overhaul.md`
- `plans/community-detection-clustering.md`
- `plans/episode-training-1000s.md`

### Region splitting

- `region-splitting/main.py`
- `region-splitting/services/registry.py`
- `region-splitting/services/base.py`
- `region-splitting/services/dbscan.py`
- `region-splitting/services/leiden.py`
- `region-splitting/services/louvian.py`
- `region-splitting/services/hybrid.py`
- `region-splitting/generate_clusters.py`
- `region-splitting/generate_dbscan_clusters.py`
- `region-splitting/tests/test_dbscan.py`

### RL and simulation

- `advesarial/src/config.py`
- `advesarial/src/sample.py`
- `advesarial/src/evaluate.py`
- `advesarial/src/services/traci.py`
- `advesarial/src/environment/environment.py`
- `advesarial/src/models/encoder/local.py`
- `advesarial/src/models/encoder/transformer.py`
- `advesarial/src/models/gat.py`
- `advesarial/src/models/lstm.py`
- `advesarial/src/agents/actor.py`
- `advesarial/src/memory/replay_buffer.py`
- `advesarial/src/utils/loss.py`

### Reference material

- `papers/HiLight_A_Hierarchical_Reinforcement_Learning_Fram (1).pdf`

---

## Appendix B. Reproducibility Checklist

Every future result intended for board, paper, or funding review should include:

- Repository commit hash.
- Scenario and SUMO network file hash.
- Demand/route file and route selection.
- Simulator version.
- Python and dependency lock information.
- Cluster method and all parameters.
- Cluster JSON hash.
- Random seed(s).
- Training episodes and steps.
- Evaluation horizon.
- Checkpoint path and checkpoint hash.
- Baseline controller and configuration.
- Completed, departed, and unfinished vehicle counts.
- Mean and percentile travel time, delay, and waiting time.
- Queue, spillback, throughput, and emissions measures.
- Fairness metrics.
- Emergency-vehicle metrics where applicable.
- Incident configuration where applicable.
- Confidence intervals over independent seeds.

This checklist is necessary to turn the current implementation from an exploratory prototype into a defensible research foundation.
