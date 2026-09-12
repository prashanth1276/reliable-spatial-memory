"""
Main experiment.

5 policies x 4 scenarios x 4 noise levels x 50 seeds = 4000 runs.

Run:  python scripts\run_main_experiment.py
"""
import os
os.environ["PYTHONHASHSEED"] = "0"

import csv
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsm.environment.grid_world import GridWorld
from rsm.environment.scene import default_house
from rsm.perception.observation import Frame, ObjectObservation
from rsm.perception.noise_model import PerceptionNoise, NoiseConfig
from rsm.memory.spatial_graph import SpatialGraph
from rsm.memory.updater import update_memory
from rsm.evaluation.scenarios import SCENARIO_FNS, ScenarioType
from rsm.evaluation.policies import POLICIES
from rsm.tasks.navigate import run_navigate_task


SEEDS = 50
EXPLORE_STEPS = 150
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.30]


def clean_frame(obs, step):
    objects = [
        ObjectObservation(
            object_id=o["objectId"], object_type=o["type"],
            x=float(o["x"]), y=float(o["y"]),
            room_id=o.get("room_id", ""),
            parent_receptacle_id=o.get("parent_id") or None,
            distance=o.get("distance"), visible=True, confidence=0.9,
        )
        for o in obs.visible_objects
    ]
    return Frame(step, obs.agent_x, obs.agent_y, obs.agent_facing, objects)


def noisy_frame(obs, step, noise, location_ids):
    objects = [
        ObjectObservation(
            object_id=o["objectId"], object_type=o["type"],
            x=float(o["x"]), y=float(o["y"]),
            room_id=o.get("room_id", ""),
            parent_receptacle_id=o.get("parent_id") or None,
            distance=o.get("distance"), visible=True, confidence=0.9,
        )
        for o in obs.visible_objects
    ]
    if noise is not None:
        objects = noise.corrupt(objects, location_ids)
    return Frame(step, obs.agent_x, obs.agent_y, obs.agent_facing, objects)


def explore(env, memory, n, seed):
    rng = random.Random(seed)
    for i in range(n):
        r = rng.random()
        action = ("TURN_LEFT" if r < 0.25 else
                  "TURN_RIGHT" if r < 0.50 else "MOVE_FORWARD")
        obs = env.step(action)
        update_memory(memory, clean_frame(obs, i + 1))


def _stable_hash(s: str) -> int:
    """Deterministic string hash — Python's built-in hash() is randomized."""
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h


def run_one(policy_name, scenario_name, noise_miss, seed):
    env = GridWorld(scene=default_house())
    memory = SpatialGraph()
    explore(env, memory, EXPLORE_STEPS, seed=seed)

    scenario = SCENARIO_FNS[scenario_name](env)

    noise = None
    if noise_miss > 0.0:
        noise = PerceptionNoise(
            NoiseConfig(
                miss_rate=noise_miss,
                position_sigma=0.30,
                wrong_parent_rate=0.08,
                uncertain_rate=0.20,
            ),
            seed=seed * 1000 + _stable_hash(policy_name) % 997,
        )

    # Inject the "noisy" scenario's bad reading into the FIRST observation
    # by pre-corrupting what the policy sees. We do this by wrapping the
    # frame builder in run_navigate_task: for the noisy scenario, force the
    # target's parent to the wrong value once.
    def task_frame_builder(obs, step, noise_model, loc_ids):
        f = noisy_frame(obs, step, noise_model, loc_ids)
        if scenario.scenario_type == ScenarioType.NOISY:
            for o in f.objects:
                if o.object_id == scenario.object_id:
                    o.parent_receptacle_id = scenario.noise_target_location
                    o.confidence = 0.85
        return f

    result = run_navigate_task(
        env=env, memory=memory,
        target_id=scenario.object_id,
        frame_builder=task_frame_builder,
        policy_fn=POLICIES[policy_name],
        scenario=scenario,
        noise=noise,
        location_ids=list(env.scene.objects.keys()),
    )

    return {
        "policy": policy_name,
        "scenario": scenario_name,
        "noise_miss": noise_miss,
        "seed": seed,
        "task_success": int(result["success"]),
        "memory_correct": result["memory_correct"],
        "attempts": result["attempts"],
        "nav_steps": result["nav_steps"],
        "conflicts": result["conflicts"],
        "reobservations": result["reobservations"],
    }


def main():
    total = len(POLICIES) * len(SCENARIO_FNS) * len(NOISE_LEVELS) * SEEDS
    print("=" * 72)
    print("MAIN EXPERIMENT")
    print(f"Policies:    {list(POLICIES.keys())}")
    print(f"Scenarios:   {list(SCENARIO_FNS.keys())}")
    print(f"Noise levels: {NOISE_LEVELS}")
    print(f"Seeds:       {SEEDS}")
    print(f"Total runs:  {total}")
    print("=" * 72)

    results = []
    for policy in POLICIES:
        for scenario in SCENARIO_FNS:
            for nl in NOISE_LEVELS:
                for seed in range(SEEDS):
                    results.append(run_one(policy, scenario, nl, seed))

    os.makedirs("results/tables", exist_ok=True)
    csv_path = "results/tables/main_experiment.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path}  ({len(results)} rows)")

    # -------- print summary at max noise --------
    max_noise = max(NOISE_LEVELS)
    agg = defaultdict(lambda: {"n": 0, "ok": 0, "mem": 0, "reob": 0})
    for r in results:
        if r["noise_miss"] != max_noise:
            continue
        k = (r["policy"], r["scenario"])
        agg[k]["n"] += 1
        agg[k]["ok"] += r["task_success"]
        agg[k]["mem"] += r["memory_correct"]
        agg[k]["reob"] += r["reobservations"]

    print("\n" + "=" * 72)
    print(f"RESULTS at noise={max_noise}")
    print("=" * 72)
    print(f"{'policy':<22} {'scenario':<16} {'task_ok':<10} "
          f"{'mem_ok':<10} {'reobs':<10}")
    print("-" * 72)
    for policy in POLICIES:
        for scenario in SCENARIO_FNS:
            a = agg[(policy, scenario)]
            print(f"{policy:<22} {scenario:<16} "
                  f"{a['ok'] / a['n']:<10.2f} "
                  f"{a['mem'] / a['n']:<10.2f} "
                  f"{a['reob'] / a['n']:<10.2f}")
    print("=" * 72)

    # -------- markdown summary --------
    md = "results/tables/main_experiment.md"
    with open(md, "w") as f:
        f.write("# Main Experiment Results\n\n")
        f.write(f"Results at noise={max_noise}.\n\n")
        f.write("| Policy | Scenario | Task Success | Memory Correct | "
                "Avg Re-observations |\n")
        f.write("|---|---|---|---|---|\n")
        for policy in POLICIES:
            for scenario in SCENARIO_FNS:
                a = agg[(policy, scenario)]
                f.write(f"| {policy} | {scenario} | "
                        f"{a['ok'] / a['n']:.2f} | "
                        f"{a['mem'] / a['n']:.2f} | "
                        f"{a['reob'] / a['n']:.2f} |\n")
    print(f"Saved: {md}")


if __name__ == "__main__":
    main()