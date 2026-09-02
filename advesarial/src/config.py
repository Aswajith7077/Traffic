import json
import os

SCENARIO = os.environ.get("TRAFFIC_SCENARIO", "manhattan")
CLUSTER_METHOD = os.environ.get("CLUSTER_METHOD", "dbscan")

# Every partition method the pipeline can produce. The path resolution is
# generic over this list so any pluggable method (Leiden, Louvain, DBSCAN and
# the hybrid combinations) can be selected purely via CLUSTER_METHOD.
VALID_METHODS = (
    "dbscan",
    "leiden",
    "louvian",
    "dbscan_louvian",
    "dbscan_leiden",
    "louvian_dbscan",
    "leiden_dbscan",
)


class Config:
    def __init__(self, path_to_cluster: str):
        self.path_to_cluster = path_to_cluster
        self.cluster_config = None
        self.__init_cluster_config()

    def __init_cluster_config(self):

        with open(self.path_to_cluster, "r") as f:
            self.cluster_config = json.load(f)

        if not self.cluster_config:
            raise ValueError("Cluster config not found")

        self.clusters = self.cluster_config["clusters"]
        self.metrics = self.cluster_config["metrics"]


def _resolve_cluster_path() -> str:
    """Resolve the cluster file for the selected scenario and method.

    Prefers ``CLUSTER_METHOD`` verbatim so an explicitly requested partition
    (including hybrid combinations) is honored even when it produces a single
    region. When the requested method's file is missing, falls back to
    ``dbscan`` and then ``leiden`` so the RL pipeline always finds a partition.
    """
    for method in (CLUSTER_METHOD, "dbscan", "leiden"):
        candidate = f"clusters/{method}/{SCENARIO}_clusters.json"
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No cluster file for scenario '{SCENARIO}' under method '{CLUSTER_METHOD}' "
        "(tried dbscan and leiden too). Run region-splitting first."
    )


config = Config(_resolve_cluster_path())
