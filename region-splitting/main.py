import json

from schema import TraciConfig
from services import LeidenService, LouvianService, TraciService


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


def perform_leiden(traci_service: TraciService, module_name: str):
    leiden_service = LeidenService(
        f"sumo/{module_name}/{module_name}.net.xml", traci_service
    )
    leiden_service.build_graph()
    clusters = leiden_service.get_clusters()
    leiden_service.generate_visualization(
        clusters["clusters"], f"visualizations/leiden/{module_name}.png"
    )

    with open(f"clusters/leiden/{module_name}_clusters.json", "w") as f:
        json.dump(clusters, f, indent=4)


def main(module_name: str):
    config = TraciConfig(config_path=f"sumo/{module_name}/{module_name}.sumocfg")
    traci_service = TraciService(config)

    traci_service.start_simulation()

    # perform_louvian(traci_service, module_name)
    perform_leiden(traci_service, module_name)


if __name__ == "__main__":
    # Only two options either "simple" or "osm"
    main("osm")
