import sys, os, torch

sys.path.insert(0, os.path.abspath("src"))

from models import GATLayer, LocalEncoder, TransformerEncoder, SubGoalGenerator
from agents import ActorCritic
from schema import TransformerEncoderConfig
from services import TraciService
from schema import TraciConfig

os.environ["SUMO_HOME"] = "C:\\Program Files (x86)\\Eclipse\\Sumo"
traci_config = TraciConfig(config_path="../scenarios/cologne8/cologne8.sumocfg")
service = TraciService(traci_config)
service.start_simulation()

tls_set = set(service.get_all_intersections())
print("TLS count:", len(tls_set))
print("TLS ids:", sorted(tls_set))

import json

clusters_raw = json.load(open("clusters/leiden/cologne8_clusters.json"))["clusters"]
clusters = {}
for cid, nodes in clusters_raw.items():
    filtered = [n for n in nodes if n in tls_set]
    if filtered:
        clusters[cid] = filtered
m = len(clusters)
print("m (clusters after TLS filter):", m)

local_encoder = LocalEncoder()
GAT = GATLayer(feature_dim=64)
actor_critic = ActorCritic(state_dimension=128, action_dimension=7)
transformer_encoder = TransformerEncoder(TransformerEncoderConfig(d_model=128, nhead=8, num_layers=6))
subgoal_generator = SubGoalGenerator(d_reg=128, d_hidden=128, M=m, d_g=2 * len(tls_set))

out = "../models/run_test_cologne8"
os.makedirs(out, exist_ok=True)
torch.save(transformer_encoder.state_dict(), f"{out}/transformer_encoder.pth")
torch.save(subgoal_generator.state_dict(), f"{out}/subgoal_generator.pth")
torch.save(local_encoder.state_dict(), f"{out}/local_encoder.pth")
torch.save(GAT.state_dict(), f"{out}/gat.pth")
torch.save(actor_critic.state_dict(), f"{out}/actor_critic.pth")
service.close_simulation()
print("saved fresh untrained cologne8-shaped model to", out)
