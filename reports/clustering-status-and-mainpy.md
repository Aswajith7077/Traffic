# Clustering Implementation Status & `main.py` vs `pipeline.py`

**Repository:** `Traffic`
**Branch:** `clustering-approaches`
**Scope:** Verify the community-detection + clustering plan was delivered, and explain the relationship between `region-splitting/main.py` and the repo-root `pipeline.py`.
**Report date:** 2026-09-10

---

## 1. Clustering implementation — status: delivered

The plan in [`plans/community-detection-clustering.md`](../plans/community-detection-clustering.md) called for four hybrid
combinations of DBSCAN with Louvain/Leiden community detection, on top of the
three existing standalone methods (`leiden`, `louvian`, `dbscan`). This has
been implemented and verified against the plan:

| Planned item | Where it lives | Verified |
|---|---|---|
| `services/registry.py` — method → service registry | `region-splitting/services/registry.py` | ✅ `ALL_METHODS` = 3 standalone + 4 hybrid |
| `services/hybrid.py` — 4 composed combinations + metrics | `region-splitting/services/hybrid.py` | ✅ `HybridClusteringService(first, second, ...)` |
| `services/dbscan.py` — auto-tuned `eps`/`min_samples` | `region-splitting/services/dbscan.py` | ✅ `auto_tune_config`, `min_samples_for`, `choose_eps` |
| `main.py` dispatch through the registry | `region-splitting/main.py:25` | ✅ `create_service(method, ...)` |
| `generate_clusters.py` — bulk generation, all methods × scenarios | `region-splitting/generate_clusters.py` | ✅ imports `main.run_partition` directly |
| `pipeline.py` — `--cluster-method`, `CLUSTER_METHOD` env | `pipeline.py:48-56`, `pipeline.py:186-193` | ✅ |
| `advesarial/src/config.py` — generic cluster path resolution | `advesarial/src/config.py:39-55` | ✅ tries `CLUSTER_METHOD` → `dbscan` → `leiden` |

Checks run for this report:

- **Unit tests:** `region-splitting/tests/test_dbscan.py` — 10/10 passed.
- **Generated artifacts:** all 7 methods (`dbscan`, `leiden`, `louvian`,
  `dbscan_leiden`, `dbscan_louvian`, `leiden_dbscan`, `louvian_dbscan`) have
  cluster JSON + visualizations for all 5 scenarios (manhattan, cologne8,
  ingolstadt21, arterial4x4, grid4x4) under both `region-splitting/clusters/`
  and `advesarial/clusters/` (the copy step's destination).
- **Metrics schema:** spot-checked
  `region-splitting/clusters/louvian_dbscan/cologne8_clusters.json` — matches
  the plan's documented schema (`order`, `cd_stage`, `dbscan_stage`,
  `n_clusters`, `coverage`, `cluster_size_statistics`, `silhouette`,
  `davies_bouldin`, `calinski_harabasz`).

One thing worth flagging, not a defect: for `cologne8`, `louvian_dbscan`
collapses to a single community (`modularity: 0.0`, `n_communities: 1`), so
the DBSCAN refinement stage has nothing to subdivide. This is the "legitimate
outcome" the plan itself calls out in section 2.2 — dense/compact networks
can produce degenerate communities — but it does mean that method contributes
no useful partition for `cologne8` specifically, and should be treated as
such when comparing methods across scenarios rather than assumed to have
failed.

**Bottom line:** the clustering work matches what was planned and is wired
through the CLI, env var, and config layers end to end. Nothing further is
needed to consider this phase complete.

---

## 2. `region-splitting/main.py` vs repo-root `pipeline.py`

These are not overlapping implementations of the same thing — they sit at
different layers.

### What each one is

- **`region-splitting/main.py`** is the CLI entrypoint *for the
  region-splitting subproject only*. Given `--scenario` and `--method`, it:
  1. builds the road graph (`service.build_graph()`),
  2. runs the selected partitioner via `create_service(method, ...)`
     (`region-splitting/main.py:25`),
  3. auto-tunes DBSCAN if needed, computes the metrics block, and
  4. writes `clusters/<method>/<scenario>_clusters.json` and a visualization
     PNG.

  It also exposes this as an importable function, `run_partition()`
  (`region-splitting/main.py:13`), which `generate_clusters.py` calls directly
  in-process to bulk-generate every method × scenario combination without
  shelling out — so `main.py` is already doing double duty as both a CLI and
  a library module, not just a CLI.

- **`pipeline.py`** (repo root) is a cross-package orchestrator. It doesn't
  contain any clustering, training, or evaluation logic itself — it shells
  out via `subprocess.run` to whichever subproject owns that step:
  - `step_cluster` → `region-splitting/main.py` (`pipeline.py:186-193`)
  - `step_train` → `advesarial/src/sample.py` (`pipeline.py:215-222`)
  - `step_eval` / `step_run_gui` → `advesarial/src/evaluate.py`
  - `step_baseline` → the `sumo` binary directly

  Each subprocess runs with `VENV_PYTHON` and a scenario/method-scoped `env`
  dict (`pipeline_env()`, `pipeline.py:73-86`), and `pipeline.py` just checks
  return codes and copies files between steps (`step_copy`,
  `pipeline.py:196-212`).

### Why they're separate, and why that's the right call

1. **Different packages, different dependencies.** `region-splitting/` and
   `advesarial/` each have their own `requirements.txt` and their own
   same-named modules (`services`, `schema`). Importing both into one Python
   process would collide; running each as its own subprocess avoids that
   entirely and is why `pipeline.py` uses `subprocess.run` instead of
   `import`.
2. **Independent usability.** Because `main.py` is a self-contained CLI, the
   region-splitting work can be run, tested, and iterated on in isolation —
   `cd region-splitting && python main.py --method leiden_dbscan` — without
   touching training or evaluation. `generate_clusters.py` also depends on
   it directly for bulk generation. Folding `main.py`'s logic into
   `pipeline.py` would break both of those uses or force `pipeline.py` to
   import cross-package code.
3. **`pipeline.py` already covers it, one layer up.** `pipeline.py --cluster-method <method>` *does* run clustering — through `main.py`, not
   instead of it. So there's no missing functionality; the question is only
   whether `main.py` needs to keep existing as a separate file, and it does,
   because `pipeline.py` calls it as a subprocess and `generate_clusters.py`
   imports it as a module.

### Recommendation

Keep `main.py` as-is. It is not redundant with `pipeline.py` — it's the
thing `pipeline.py`'s cluster step and `generate_clusters.py`'s bulk-generate
both depend on. Collapsing it into `pipeline.py` would require either (a)
cross-importing `region-splitting` and `advesarial` modules into one process
(dependency/namespace collisions), or (b) inlining the partitioning logic
into the root-level orchestrator, which would break `generate_clusters.py`
and make the region-splitting subproject non-runnable on its own. No action
needed here.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
