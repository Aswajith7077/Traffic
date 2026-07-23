import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
else:
    sys.exit("Environment variable SUMO_HOME not declared")

import traci

sys.path.insert(0, "src")
from schema import TraciConfig
from services import TraciService

config = TraciConfig(config_path="sumo/osm.sumocfg", use_gui=True, delay=50)
service = TraciService(config)
service.start_simulation()

try:
    while True:
        service.step()
except (KeyboardInterrupt, traci.exceptions.FatalTraCIError):
    service.close_simulation()
