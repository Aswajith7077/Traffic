"""Paper-style ATT/ADT computation from SUMO tripinfo output.

Matches the HiLight definitions:
  ATT = mean over vehicles of (exit_time - entry_time)
  ADT = mean over vehicles of the delay vs. free-flow travel
SUMO's tripinfo `<duration>` attribute is (exit - entry); the
`<timeLoss>` attribute is SUMO's accumulated delay, used as ADT.
"""

import os
import time
import xml.etree.ElementTree as ET


def wait_for_tripinfo(path, timeout=30.0):
    """Wait until SUMO has finished writing the tripinfo file (it is flushed
    asynchronously when the simulation process exits)."""
    deadline = time.time() + timeout
    last_size = -1
    while time.time() < deadline:
        if os.path.exists(path):
            with open(path, "rb") as f:
                size = os.path.getsize(path)
            if size == last_size:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        f.seek(max(size - 128, 0))
                        tail = f.read()
                    if "</tripinfos>" in tail:
                        return True
                except OSError:
                    pass
            last_size = size
        time.sleep(0.5)
    return False


def parse_tripinfo(path):
    last_err = None
    for _ in range(12):  # retry up to ~60s; SUMO flushes tripinfo asynchronously on exit
        try:
            tree = ET.parse(path)
            break
        except (ET.ParseError, OSError) as e:
            last_err = e
            time.sleep(5)
    else:
        print(f"WARNING: tripinfo at {path} still unreadable: {last_err}")
        tree = ET.parse(path)
    root = tree.getroot()

    durations = []
    time_losses = []
    waiting_times = []
    teleports = 0
    total = 0

    for ti in root.findall("tripinfo"):
        total += 1
        if ti.attrib.get("teleport") not in (None, "0"):
            teleports += 1

        try:
            duration = float(ti.attrib.get("duration", -1))
        except ValueError:
            duration = -1

        if duration < 0 or ti.attrib.get("arrival", "").startswith("-"):
            continue  # vehicle still running at simulation end

        durations.append(duration)
        time_losses.append(float(ti.attrib.get("timeLoss", 0.0)))
        waiting_times.append(float(ti.attrib.get("waitingTime", 0.0)))

    n = len(durations)
    if n < total:
        print(
            f"WARNING: only {n}/{total} vehicles have arrivals in tripinfo "
            f"(sim may have ended while vehicles were still running)"
        )
    return {
        "vehicles_departed": total,
        "vehicles_ended": n,
        "teleports": teleports,
        "att": sum(durations) / n if n else 0.0,
        "adt": sum(time_losses) / n if n else 0.0,
        "avg_wait": sum(waiting_times) / n if n else 0.0,
    }


def print_tripinfo_summary(metrics):
    print("\n" + "=" * 50)
    print("PAPER-STYLE TRIPINFO METRICS (HiLight definition)")
    print("=" * 50)
    print(f"Vehicles Departed        : {metrics['vehicles_departed']}")
    print(f"Vehicles with Arrival    : {metrics['vehicles_ended']}")
    print(f"Teleported Vehicles      : {metrics['teleports']}")
    print(f"Average Travel Time ATT  : {metrics['att']:.2f} seconds")
    print(f"Average Delay Time ADT   : {metrics['adt']:.2f} seconds")
    print(f"Average Waiting Time     : {metrics['avg_wait']:.2f} seconds")
    print("=" * 50)
