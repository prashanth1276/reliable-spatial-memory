"""
Correlated noise experiment.

Same design as run_main_experiment.py, but observations suffer from
temporally correlated noise (OU process) instead of independent noise.

Run:  python scripts\run_correlated_experiment.py
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
from rsm.perception.correlated_noise import CorrelatedNoise, CorrelatedNoiseConfig
from rsm.memory.spatial_graph import SpatialGraph
from rsm.memory.updater import update_memory
from rsm.evaluation.scenarios import SCENARIO_FNS, ScenarioType
from rsm.evaluation.policies import POLICIES
from rsm.tasks.navigate import run_navigate_task


SEEDS = 30
EXPLORE_STEPS = 150
CORRELATION_LEVELS = [0.0, 0.3, 0.6, 0.9]


def _stable_hash(s: str) -> int:
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h


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


def run_one(policy_name, scenario_name, corr_strength, seed):
    env = GridWorld(scene=default_house())
    memory = SpatialGraph()
    explore(env, memory, EXPLORE_STEPS, seed=seed)

    scenario = SCENARIO_FNS[scenario_name](env)

    noise = CorrelatedNoise(
        CorrelatedNoiseConfig(
            miss_rate=0.12,
            position_sigma=0.30,
            wrong_parent_rate=0.08,
            uncertain_rate=0.20,
            correlation_time=4.0,
            correlation_strength=corr_strength,
        ),
        seed=seed * 1000 + _stable_hash(policy_name) % 997,
    )

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
        "correlation": corr_strength,
        "seed": seed,
        "task_success": result["success"],
        "memory_correct": result["memory_correct"],
        "reobservations": result["reobservations"],
    }


def main():
    total = len(POLICIES) * len(SCENARIO_FNS) * len(CORRELATION_LEVELS) * SEEDS
    print("=" * 72)
    print("CORRELATED NOISE EXPERIMENT")
    print(f"Policies: {list(POLICIES.keys())}")
    print(f"Scenarios: {list(SCENARIO_FNS.keys())}")
    print(f"Correlation strengths: {CORRELATION_LEVELS}")
    print(f"Seeds: {SEEDS} | Total runs: {total}")
    print("=" * 72)

    results = []
    for policy in POLICIES:
        for scenario in SCENARIO_FNS:
            for cs in CORRELATION_LEVELS:
                for seed in range(SEEDS):
                    results.append(run_one(policy, scenario, cs, seed))

    os.makedirs("results/tables", exist_ok=True)
    csv_path = "results/tables/correlated_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path} ({len(results)} rows)")

    max_corr = max(CORRELATION_LEVELS)
    agg = defaultdict(lambda: {"n": 0, "ok": 0, "mem": 0, "reob": 0})
    for r in results:
        if r["correlation"] != max_corr:
            continue
        k = (r["policy"], r["scenario"])
        agg[k]["n"] += 1
        agg[k]["ok"] += r["task_success"]
        agg[k]["mem"] += r["memory_correct"]
        agg[k]["reob"] += r["reobservations"]

    print("\n" + "=" * 72)
    print(f"RESULTS at correlation={max_corr}")
    print("=" * 72)
    print(f"{'policy':<22} {'scenario':<16} {'task_ok':<10} {'mem_ok':<10} {'reobs':<10}")
    print("-" * 72)
    for policy in POLICIES:
        for scenario in SCENARIO_FNS:
            a = agg[(policy, scenario)]
            print(f"{policy:<22} {scenario:<16} "
                  f"{a['ok'] / a['n']:<10.2f} {a['mem'] / a['n']:<10.2f} "
                  f"{a['reob'] / a['n']:<10.2f}")
    print("=" * 72)

    md = "results/tables/correlated_results.md"
    with open(md, "w") as f:
        f.write("# Correlated Noise Experiment\n\n")
        f.write(f"Results at correlation strength = {max_corr}.\n\n")
        f.write("| Policy | Scenario | Task Success | Memory Correct | Re-observations |\n")
        f.write("|---|---|---|---|---|\n")
        for policy in POLICIES:
            for scenario in SCENARIO_FNS:
                a = agg[(policy, scenario)]
                f.write(f"| {policy} | {scenario} | "
                        f"{a['ok'] / a['n']:.2f} | {a['mem'] / a['n']:.2f} | "
                        f"{a['reob'] / a['n']:.2f} |\n")
    print(f"Saved: {md}")


if __name__ == "__main__":
    main()