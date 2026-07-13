from abc import ABC
from abc import abstractmethod
from .traci import TraciService
import sumolib


class BaseClusteringService(ABC):
    def __init__(self, net_config_path: str, traci_service: TraciService):
        self.net_config_path = net_config_path
        self.traci_service = traci_service
        self.net = sumolib.net.readNet(self.net_config_path)

    @abstractmethod
    def build_graph(self):
        pass

    @abstractmethod
    def get_clusters(self):
        pass

    @abstractmethod
    def generate_visualization(self, clusters: dict, output_path: str):
        pass
