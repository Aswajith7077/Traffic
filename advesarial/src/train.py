from agents import Actor
from config import Config
from environment import Environment
from memory import ReplayBuffer
from schema import ReplayBufferItem, TraciConfig


def train(T: int = 100):

    environment = Environment(config=TraciConfig(config_path="sumo/manhattan.sumocfg"))
    config = Config("clusters/louvian/manhattan_clusters.json")
    replay_buffer = ReplayBuffer(max_size=1000)
    environment.reset()

    for t in range(T):
        state = environment.get_state()
        action = Actor(state)
        next_state, reward, done = environment.step(action=action)

        buffer_item = ReplayBufferItem(state=state, action=action, reward=reward, next_state=next_state, done=done)
        replay_buffer.add(buffer_item)

        pass

    print(config.clusters)
    print(config.metrics)


def modules():

    environment = Environment(config=TraciConfig(config_path="sumo/manhattan.sumocfg"))

    # traci_config = TraciConfig(config_path="sumo/manhattan.sumocfg")
    # traci_service = TraciService(traci_config)

    # # Initialize models
    # transformer_encoder = TransformerEncoder(TransformerEncoderConfig())
    # sub_goal_generator = SubGoalGenerator()
    # sub_policy = SubPolicy()


config = Config("clusters/louvian/manhattan_clusters.json")
environment = Environment(
    config=config,
    traci_config=TraciConfig(config_path="sumo/manhattan.sumocfg"),
)
state = environment.get_state()

print(state.shape)
