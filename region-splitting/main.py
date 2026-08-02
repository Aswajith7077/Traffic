from schema import TraciConfig
from services import TraciService
from services import LouvianService
from services import LeidenService

import json


def perform_louvian(traci_service: TraciService, module_name: str):

    louvian_service = LouvianService(
        f"sumo/{module_name}/{module_name}.net.xml", traci_service
    )
    louvian_service.build_graph()
    clusters = louvian_service.get_clusters()
    louvian_service.generate_visualization(
        clusters["clusters"], f"visualizations/louvian/{module_name}.png"
    )

    with open(f"clusters/louvian/{module_name}_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)


def perform_leiden(
    traci_service: TraciService, network_path: str, scenario_name: str
):
    leiden_service = LeidenService(
        network_path, traci_service
    )
    leiden_service.build_graph()
    clusters = leiden_service.get_clusters()
    leiden_service.generate_visualization(
        clusters["clusters"], f"visualizations/leiden/{scenario_name}.png"
    )

    with open(f"clusters/leiden/{scenario_name}_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)


def main():
    config = TraciConfig(config_path="../scenarios/manhattan/manhattan.sumocfg")
    traci_service = TraciService(config)

    traci_service.start_simulation()

    perform_leiden(
        traci_service,
        "../scenarios/manhattan/manhattan.net.xml",
        "manhattan",
    )


if __name__ == "__main__":
    main()
