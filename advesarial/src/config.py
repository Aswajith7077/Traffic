import json


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


config = Config("clusters/louvian/manhattan_clusters.json")
