# Fix: Clustering/Community Detection Considers All Junctions, Not Just Traffic-Light Intersections

## 1. Verdict

**Confirmed bug.** The clustering and community-detection stage
(`region-splitting/services/{leiden,louvian,dbscan,hybrid}.py`) builds its graph /
point cloud from **every junction node in the SUMO net** (`self.net.getNodes()`),
regardless of whether that junction has a traffic light. The HiLight paper's
regional partitioning is defined over traffic-signal intersections only. This is
the "over-consideration" the user suspected — confirmed, not hypothetical.

Downstream, `advesarial/src/training/trainer.py:57-61` *does* filter each cluster
down to traffic-light nodes before computing `M` and running the RL loop:

```python
raw_clusters = config.clusters
tls_set = set(self.intersections)  # traci.trafficlight.getIDList()
self.clusters = {cid: [n for n in nodes if n in tls_set] for cid, nodes in raw_clusters.items()}
self.clusters = {cid: nodes for cid, nodes in self.clusters.items() if nodes}
self.M = len(self.clusters)
```

So the RL agent set itself is correct (traffic lights only). The bug is upstream:
**the partition boundaries, quality metrics, and region assignment were computed
on a graph/point-cloud dominated by non-signalized junctions**, then merely
stripped of non-TLS members after the fact. Filtering members out of an
already-formed cluster is not the same as clustering the right population in the
first place — see §3 for why this actually changes results, not just node counts.

## 2. Evidence

### 2.1 Where the over-consideration happens

| File | Line(s) | What it does |
|---|---|---|
| `region-splitting/services/leiden.py` | `build_graph()` (L17-45) | Adds a vertex for every `edge.getFromNode()/getToNode()` in the net — all junction types |
| `region-splitting/services/louvian.py` | `build_graph()` (L24-35) | Same — `self.graph.add_edge(source, destination, ...)` over all nodes |
| `region-splitting/services/dbscan.py` | `DBSCANService.build_graph()` (L878-897) | `for node in self.net.getNodes(): ...` — point cloud is every junction's coordinate |
| `region-splitting/services/hybrid.py` | `HybridClusteringService.build_graph()` (L105-125) | Same pattern — feeds all 4 hybrid combinations (`dbscan_louvian`, `dbscan_leiden`, `louvian_dbscan`, `leiden_dbscan`) |
| `region-splitting/main.py` | `run_partition()` (L38) | `total_net_nodes = len(service.net.getNodes())` — the `coverage` metric is measured against all junctions, not traffic lights, so it's currently ~vacuous (always ≈1.0) instead of meaningful |

None of these four services filter by `node.getType()` or cross-check against
`net.getTrafficLights()` anywhere. No TLS-aware filtering exists before the
partitioning algorithms run.

### 2.2 How lopsided this is, per scenario

Measured directly against each scenario's `.net.xml` (`node.getType()` breakdown
and `net.getTrafficLights()`, the same authoritative source
`advesarial/src/services/traci.py::get_all_intersections()` uses via
`traci.trafficlight.getIDList()` at simulation time):

| Scenario | Total junctions | Traffic-light junctions | TLS share | Non-TLS types present |
|---|---:|---:|---:|---|
| `manhattan` | 2774 | 472 | 17.0% | `priority` (516), `dead_end` (1781), `right_before_left` (5) |
| `cologne8` | 78 | 8 | 10.3% | `right_before_left` (15), `priority` (50), `dead_end` (5) |
| `ingolstadt21` | 381 | 21 (24 nodes typed `traffic_light`, but 3 have no controllable program — `net.getTrafficLights()` is the correct authoritative count) | 5.5–6.3% | `priority` (281), `dead_end` (34), `right_before_left` (41), `zipper` (1) |
| `arterial4x4` | 32 | 16 | 50.0% | `priority` (16) |
| `grid4x4` | 32 | 16 | 50.0% | `dead_end` (16) — note: type is `traffic_light_right_on_red`, not plain `traffic_light` |

For the three real/OSM-derived networks (`manhattan`, `cologne8`,
`ingolstadt21`) — the scenarios that matter most for the paper comparison —
**83–94% of the nodes driving the clustering algorithm are not traffic signals
at all.** `arterial4x4`/`grid4x4` are synthetic 4×4 grids where every junction
happens to be signalized-ish, so they're the two scenarios where this bug is
mostly harmless by coincidence.

Also worth flagging: `node.getType() == "traffic_light"` string-matching is
itself not fully reliable (ingolstadt21: 24 typed nodes vs. 21 actually in
`net.getTrafficLights()`; grid4x4 uses the type string
`traffic_light_right_on_red`, not `traffic_light`). The fix must use
`net.getTrafficLights()` (or equivalently `traci.trafficlight.getIDList()` when
a live simulation is available), matching what the RL side already trusts —
not a type-string comparison.

## 3. Why this is a real bug, not just harmless extra bookkeeping

`trainer.py`'s post-hoc filter fixes the *agent* population (M and which node
belongs to which region-id are eventually TLS-only), but it cannot undo damage
already baked into the partition *shape*:

1. **Community detection topology is distorted.** Louvain/Leiden modularity
   optimizes over the full graph, where two traffic lights that are directly
   connected in reality (via a chain of unsignalized priority/dead-end nodes)
   look like they're several hops apart, each hop contributing its own
   vertex/edge to the modularity calculation. Communities can fracture along
   non-TLS junction boundaries that have no meaning for the RL problem, and the
   reported `modularity`/`cluster_quality` metrics describe a graph the RL
   system never actually uses.
2. **DBSCAN density estimation is skewed.** `eps`/`min_samples` auto-tuning
   (`auto_tune_config`) and the resulting cell shapes are fit to the density of
   *all* junctions (dominated by dead-ends and minor priority junctions in
   `manhattan`/`ingolstadt21`), not the density of traffic signals. Two
   networks with identical traffic-light layouts but different amounts of
   OSM-derived minor-junction clutter would get different DBSCAN partitions.
3. **Post-filter can silently shrink `M` and unbalance regions.** If a
   detected community/cell contains 6 non-TLS nodes and 1 TLS node,
   `trainer.py`'s filter reduces it to a singleton region — a valid but
   arbitrary outcome of where the majority-non-TLS blob happened to land, not
   a deliberate regional decision. Some clusters can also disappear entirely
   (all-non-TLS cluster → filtered to empty → dropped), meaning `M` reported in
   the JSON metrics (`n_clusters`) does not equal the `M` actually used to size
   `SubGoalGenerator` in `trainer.py`. The two numbers silently diverge today.
4. **`coverage` and clustering quality metrics in the emitted JSON are
   measuring the wrong population** (§2.1's `main.py` finding) — they can't be
   trusted as a signal of partition quality for the RL problem.

## 4. Proposed fix

### 4.1 Single source of truth for "is this a traffic signal"

Add one helper, used by every service, instead of five different informal
checks:

```python
# region-splitting/services/topology.py (new)
def get_tls_node_ids(net: sumolib.net.Net) -> set[str]:
    """Authoritative traffic-signal junction ids for a SUMO net.

    Matches advesarial/src/services/traci.py::get_all_intersections()
    (traci.trafficlight.getIDList()) so the same node population is used both
    at clustering time and at training time. Uses net.getTrafficLights()
    (nodes with a defined tlLogic program) rather than node.getType() ==
    "traffic_light", since node type strings are unreliable (see plan doc
    §2.2 — e.g. grid4x4 uses "traffic_light_right_on_red";
    ingolstadt21 has 3 traffic_light-typed nodes with no program).
    """
    return {tls.getID() for tls in net.getTrafficLights()}
```

### 4.2 Louvain/Leiden — build a TLS-projected graph, not a raw one

Non-TLS junctions can't simply be dropped from the graph — many are the only
path connecting two real traffic lights (a chain of `dead_end`/`priority`
nodes along one physical street). Dropping them naively would disconnect the
graph. Instead, **contract** each maximal run of non-TLS nodes between two TLS
nodes into a single edge:

```python
# region-splitting/services/topology.py (new)
def build_tls_projected_graph(net, tls_ids, edge_weight_fn) -> nx.Graph:
    """Undirected graph whose vertices are exactly `tls_ids`. An edge between
    two TLS nodes exists iff there is a path between them in the road network
    that passes through no other TLS node; its weight is the sum of
    edge_weight_fn(edge) along every such path (parallel paths accumulate).
    """
```

Implementation approach: build the full node/edge graph once (as today), then
for each TLS node run a bounded BFS/DFS outward, stopping whenever another TLS
node is reached (do not continue through it), summing `edge_weight_fn` along
the way; add one aggregated edge per (TLS, TLS) pair reached. This is the
standard "junction reduction" / graph contraction technique — equivalent to
what `netconvert --junctions.join` does for geometry-only nodes, generalized
to any non-TLS node type here.

- `LeidenService.build_graph()` / `LouvianService.build_graph()`: replace the
  `for edge in self.net.getEdges(): vertices.add(...)` loop with
  `build_tls_projected_graph(self.net, get_tls_node_ids(self.net), weight_fn)`,
  where `weight_fn` stays the existing simulation-derived edge weight
  (`compute_weights`) for Leiden/Louvain, or lane-count for Louvain's static
  fallback.
- `merge_singletons` / `merge_small_clusters` and `compute_cluster_quality` /
  `cluster_quality` need no change — they already operate generically over
  whatever graph they're handed.

### 4.3 DBSCAN — restrict the point cloud to TLS coordinates

Simpler: no contraction needed since DBSCAN doesn't consume graph edges for
clustering itself (only for optional visualization/adjacency).

- `DBSCANService.build_graph()`: change `for node in self.net.getNodes():` to
  `for node in self.net.getNodes(): if node.getID() not in tls_ids: continue`
  (or iterate `tls_ids` directly via `self.net.getNode(nid)`), using the same
  `get_tls_node_ids()` helper. Keep the full graph available separately if
  edges are still wanted for visualization context (draw all roads, color only
  TLS nodes) — visualization fidelity is a nice-to-have, not required for
  correctness.
- Same change in `HybridClusteringService.build_graph()` (L105-125) — this
  fixes the point cloud for both hybrid orders, *and* fixes the base graph
  used by the CD stage in `louvian_dbscan`/`leiden_dbscan` (§4.4).

### 4.4 Hybrid combinations — inherits from 4.2 + 4.3

All four hybrid methods (`dbscan_louvian`, `dbscan_leiden`, `louvian_dbscan`,
`leiden_dbscan`) go through `HybridClusteringService.build_graph()` /
`self.graph` for their community-detection stage and `self.coords` for their
DBSCAN stage. Fixing `build_graph()` per §4.2+4.3 (TLS-projected graph +
TLS-only coordinates) fixes all four combinations from one place — no
per-combination changes needed in `_supernode_graph`, `_dbscan`, or the
CD-then-DBSCAN path, since they all consume `self.graph`/`self.coords`
downstream of `build_graph()`.

### 4.5 `main.py` — fix the coverage metric

```python
# before
total_net_nodes = len(service.net.getNodes())
# after
total_net_nodes = len(get_tls_node_ids(service.net))
```

So `coverage` actually means "fraction of traffic-signal junctions assigned to
a region" (should become 1.0 once the fix lands, meaningfully, not
coincidentally).

### 4.6 `trainer.py` — keep the filter, downgrade its role

Leave `trainer.py:57-61`'s TLS filter in place as a defensive no-op / sanity
assertion (cheap insurance against a stale or hand-edited cluster JSON) rather
than removing it — but it should no longer be doing real work once §4.1-4.5
land. Optionally add an assertion/log if it ever actually removes a node post-fix,
since that would indicate the cluster JSON and the live scenario's TLS set have
drifted (e.g., regenerated net file, stale cluster cache).

## 5. Regeneration & migration

1. Implement §4.1-4.5 in `region-splitting/services/`.
2. Regenerate every cluster JSON for every method × scenario:
   `cd region-splitting && python generate_clusters.py`.
3. Re-run `pipeline.py cluster && pipeline.py copy` (or equivalent) per
   scenario/method to refresh `advesarial/clusters/**` and
   `scenarios/manhattan/manhattan_clusters.json` /
   `region-splitting/sumo/manhattan/manhattan_clusters.json` mirrors so
   nothing stale lingers.
4. Regenerate `region-splitting/visualizations/**` for a visual sanity check
   (regions should now hug traffic-light clusters, not sprawl across dead-end
   cul-de-sacs).
5. Expect `M` (number of regions) to change per scenario/method — recheck the
   `n_clusters`/`coverage`/`modularity`/`silhouette` metrics against the old
   values in `region-splitting/compare_methods.py` output to confirm the shift
   is directionally sane (fewer, more traffic-light-dense regions).
6. **Retraining is required** wherever `M` changes, since
   `SubGoalGenerator(M=self.M, ...)` and the Transformer's regional-state
   input dimension are sized from `M` — old checkpoints won't load against a
   changed `M`. This mainly affects `manhattan`, `cologne8`, `ingolstadt21`;
   `arterial4x4`/`grid4x4` should see little to no change (all/most junctions
   are already TLS-typed).

## 6. Validation plan

- **Unit-level:** for each scenario, `len(get_tls_node_ids(net))` should equal
  `len(traci.trafficlight.getIDList())` from a live simulation (cross-check
  the static sumolib helper against the runtime source of truth already used
  in `advesarial/src/services/traci.py`).
- **Coverage:** every generated cluster JSON's `clusters` values should union
  to exactly the TLS id set, no more, no less (`coverage == 1.0`,
  no non-TLS ids present).
- **Connectivity:** `build_tls_projected_graph` should produce a connected
  graph (or exactly as many components as the underlying road network has)
  for every scenario — verifies the BFS contraction didn't drop reachable TLS
  pairs.
- **`M` consistency:** `len(config.clusters)` (from the regenerated JSON)
  should equal `trainer.self.M` after the post-hoc filter — the filter should
  become a true no-op (no nodes removed) once the fix lands.
- **Smoke test:** `train_eval_cologne8.py --cluster-method louvian_dbscan
  --episodes 2 --episode-steps 200 --steps 200` (or similar short run) for
  1-2 methods to confirm the trainer boots with the new `M` and produces
  finite losses, per the same smoke-test bar used in
  `HILIGHT_SPEC_ALIGNMENT_REPORT.md` §0.

## 7. Files touched

| File | Change |
|---|---|
| `region-splitting/services/topology.py` | **new** — `get_tls_node_ids()`, `build_tls_projected_graph()` |
| `region-splitting/services/leiden.py` | `build_graph()` uses TLS-projected graph |
| `region-splitting/services/louvian.py` | `build_graph()` uses TLS-projected graph |
| `region-splitting/services/dbscan.py` | `DBSCANService.build_graph()` restricted to TLS coordinates |
| `region-splitting/services/hybrid.py` | `HybridClusteringService.build_graph()` restricted to TLS-projected graph + TLS coordinates |
| `region-splitting/main.py` | `coverage` denominator uses `get_tls_node_ids()` instead of `net.getNodes()` |
| `advesarial/src/training/trainer.py` | no functional change; filter becomes a documented no-op safety net |
| `region-splitting/clusters/**`, `advesarial/clusters/**`, `scenarios/manhattan/manhattan_clusters.json`, `region-splitting/sumo/manhattan/manhattan_clusters.json` | regenerated data, not hand-edited |
| `region-splitting/visualizations/**` | regenerated |

## 8. Open questions for the user before implementation

- For the BFS contraction in §4.2, when a chain between two TLS nodes forks
  (multiple non-TLS paths connect the same TLS pair), should edge weight be
  **summed** (more parallel capacity = stronger tie) or **averaged**? Summing
  matches how `compute_cluster_quality`'s internal/external ratio already
  treats multi-edge weight today (it sums per-edge weight), so summing is the
  consistent default — flag if a different convention is wanted.
- Should the now-regenerated cluster JSONs replace the existing ones in place
  (git history preserves the old versions), or should the old TLS+non-TLS
  JSONs be kept side-by-side (e.g. under `clusters/<method>_legacy/`) for a
  before/after comparison in the next training run's write-up?
