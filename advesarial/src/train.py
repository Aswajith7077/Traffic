from config import Config
from environment import Environment
from memory import ReplayBuffer
from schema import ReplayBufferItem
from agents import Actor
from schema import TraciConfig


def train(T: int = 100):

    environment = Environment(config=TraciConfig(config_path="sumo/osm.sumocfg"))
    config = Config("clusters/louvian/osm_clusters.json")
    replay_buffer = ReplayBuffer(max_size=1000)
    environment.reset()

    for t in range(T):
        state = environment.get_state()
        action = Actor(state)
        next_state, reward, done = environment.step(action=action)

        buffer_item = ReplayBufferItem(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )
        replay_buffer.add(buffer_item)

        pass

    print(config.clusters)
    print(config.metrics)


def modules():

    environment = Environment(config=TraciConfig(config_path="sumo/osm.sumocfg"))

    # traci_config = TraciConfig(config_path="sumo/osm.sumocfg")
    # traci_service = TraciService(traci_config)

    # # Initialize models
    # transformer_encoder = TransformerEncoder(TransformerEncoderConfig())
    # sub_goal_generator = SubGoalGenerator()
    # sub_policy = SubPolicy()


config = Config("clusters/louvian/osm_clusters.json")
environment = Environment(
    config=config,
    traci_config=TraciConfig(config_path="sumo/osm.sumocfg"),
)
state = environment.get_state()

print(state.shape)
