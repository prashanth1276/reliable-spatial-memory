"""Full experiment: 4 policies × 13 commands × 4 scenarios × 10 seeds."""
import csv
import os
import copy
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rsm.environment.grid_world import GridWorld
from rsm.environment.scene import default_house
from rsm.memory.spatial_graph import SpatialGraph
from rsm.memory.updater import update_memory
from rsm.perception.observation import Frame, ObjectObservation
from rsm.perception.noise_model import PerceptionNoise, NoiseConfig
from rsm.language.grounder import CLIPGrounder
from rsm.language.language_agent import LanguageAgent
from rsm.language.evaluation.commands import default_commands
from rsm.language.evaluation.policies import POLICIES


SEEDS = 10
# "disappeared" is excluded from the language experiment because the
# language agent has no mechanism to report "object is gone" — it always
# tries to ground a candidate. Verification handles disappearance in the
# memory layer (Project 1), not the language layer.
SCENARIO_PLAN = ["stable", "moved", "noisy"]

# Scan from the counter, facing west toward the table. This position sees:
SCAN_POS = (1, 1, 1)


def explore(env, memory, n=150, seed=42):
    rng = random.Random(seed)
    for i in range(n):
        r = rng.random()
        a = "TURN_LEFT" if r < 0.25 else "TURN_RIGHT" if r < 0.5 else "MOVE_FORWARD"
        obs = env.step(a)
        objs = [
            ObjectObservation(
                object_id=o["objectId"], object_type=o["type"],
                x=float(o["x"]), y=float(o["y"]),
                room_id=o.get("room_id", ""),
                parent_receptacle_id=o.get("parent_id") or None,
                distance=o.get("distance"), visible=True, confidence=0.9,
            ) for o in obs.visible_objects
        ]
        update_memory(memory, Frame(i + 1, obs.agent_x, obs.agent_y,
                                    obs.agent_facing, objs))


def apply_scenario(env, name):
    """Modify the world according to the scenario."""
    if name == "stable":
        return
    if name == "moved":
        c = env.get_object("counter_1")
        env.move_object("mug_1", c.x, c.y, parent_id="counter_1")
        return
    if name == "disappeared":
        env.remove_object("mug_1")
        return
    if name == "noisy":
        return  # handled via noise model, no world change


def ground_truth_for(cmd, env, scenario_name):
    """
    Which object SHOULD be chosen for this command given the current
    world state?
    Returns object_id, or None if the object is gone.
    """
    oid = cmd.ground_truth_object_id
    # For all our test commands, the target is either mug_1 or apple_1.
    # If the target no longer exists in env, ground truth is "no object".
    if env.get_object(oid) is None:
        return None
    return oid


def restrict_memory_to_visible(agent, env):
    """For no_memory policy: wipe memory, rebuild from current observation."""
    agent.memory = SpatialGraph()
    obs = env.observe()
    objects = [
        ObjectObservation(
            object_id=o["objectId"], object_type=o["type"],
            x=float(o["x"]), y=float(o["y"]),
            room_id=o.get("room_id", ""),
            parent_receptacle_id=o.get("parent_id") or None,
            visible=True, confidence=0.9,
        )
        for o in obs.visible_objects
    ]
    update_memory(agent.memory, Frame(1, obs.agent_x, obs.agent_y,
                                       obs.agent_facing, objects))


def main():
    print("=" * 72)
    print("LANGUAGE-GROUNDED NAVIGATION — EXPERIMENT")
    print(f"Policies:  {list(POLICIES)}")
    print(f"Commands:  {len(default_commands())}")
    print(f"Scenarios: {SCENARIO_PLAN}")
    print(f"Seeds:     {SEEDS}")
    print(f"Total:     "
          f"{len(POLICIES) * len(SCENARIO_PLAN) * len(default_commands()) * SEEDS} runs")
    print("=" * 72)

    print("\nLoading CLIP...")
    grounder = CLIPGrounder()

    commands = default_commands()
    results = []

    for policy in POLICIES:
        for seed in range(SEEDS):
            for scenario_name in SCENARIO_PLAN:
                # ---- Set up world + memory ----
                env = GridWorld(scene=default_house())
                memory = SpatialGraph()
                explore(env, memory, n=150, seed=seed)

                # ---- Apply scenario ----
                apply_scenario(env, scenario_name)

                # ---- Noise model for "noisy" scenario ----
                # Moderate noise: enough to trigger occasional false conflicts,
                # not so much that the vote cannot resolve them.
                noise = None
                if scenario_name == "noisy":
                    noise = PerceptionNoise(
                        NoiseConfig(
                            miss_rate=0.20,
                            position_sigma=0.20,
                            wrong_parent_rate=0.10,
                            uncertain_rate=0.15,
                            false_positive_rate=0.00,
                        ),
                        seed=seed * 100 + 42,
                    )

                # ---- Build agent ----
                memory_snapshot = copy.deepcopy(memory)

                agent = LanguageAgent(
                    env=env, memory=memory, grounder=grounder,
                    noise=noise,
                    use_memory_verification=(policy != "no_verification"),
                )
                if policy == "no_clip":
                    agent.grounder = None

                # ---- Run each command with a fresh memory ----
                for cmd in commands:
                    # Restore memory to the post-exploration snapshot
                    agent.memory = copy.deepcopy(memory_snapshot)
                    # Reset agent to scan position
                    env.agent.x, env.agent.y, env.agent.facing = SCAN_POS

                    # For no_memory policy, rebuild memory from current view
                    if policy == "no_memory":
                        restrict_memory_to_visible(agent, env)

                    gt = ground_truth_for(cmd, env, scenario_name)

                    r = agent.execute(cmd.text, gt)

                    results.append({
                        "policy": policy,
                        "scenario": scenario_name,
                        "seed": seed,
                        "command": cmd.text,
                        "category": cmd.note,
                        "parse_ok": int(r.parsed.parse_succeeded),
                        "grounding_ok": int(r.grounding_correct),
                        "nav_ok": int(r.navigation_success),
                        "end_to_end": int(r.grounding_correct
                                          and r.navigation_success),
                        "verif_fired": int(r.verification_fired),
                        "reobs": r.reobservations,
                        "chosen": r.chosen_object or "NONE",
                    })

    # ---- Save ----
    os.makedirs("results/tables", exist_ok=True)
    csv_path = "results/tables/language_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {csv_path} ({len(results)} rows)")

    # ---- Aggregate by policy ----
    agg = defaultdict(lambda: {"n": 0, "parse": 0, "ground": 0,
                                "nav": 0, "e2e": 0, "verif": 0, "reobs": 0})
    for r in results:
        k = r["policy"]
        agg[k]["n"] += 1
        agg[k]["parse"] += r["parse_ok"]
        agg[k]["ground"] += r["grounding_ok"]
        agg[k]["nav"] += r["nav_ok"]
        agg[k]["e2e"] += r["end_to_end"]
        agg[k]["verif"] += r["verif_fired"]
        agg[k]["reobs"] += r["reobs"]

    print("\n" + "=" * 72)
    print("AGGREGATED RESULTS — by policy")
    print("=" * 72)
    print(f"{'policy':<20} {'n':<6} {'parse':<8} {'ground':<8} "
          f"{'nav':<8} {'e2e':<8} {'verif':<8}")
    print("-" * 72)
    for policy in POLICIES:
        a = agg[policy]
        print(f"{policy:<20} {a['n']:<6} "
              f"{a['parse']/a['n']:<8.2f} {a['ground']/a['n']:<8.2f} "
              f"{a['nav']/a['n']:<8.2f} {a['e2e']/a['n']:<8.2f} "
              f"{a['verif']/a['n']:<8.2f}")
    print("=" * 72)

    # ---- Aggregate by (policy, scenario) ----
    agg2 = defaultdict(lambda: {"n": 0, "e2e": 0, "verif": 0})
    for r in results:
        k = (r["policy"], r["scenario"])
        agg2[k]["n"] += 1
        agg2[k]["e2e"] += r["end_to_end"]
        agg2[k]["verif"] += r["verif_fired"]

    print("\n" + "=" * 72)
    print("END-TO-END SUCCESS by (policy, scenario)")
    print("=" * 72)
    header = f"{'policy':<20} " + " ".join(f"{s:<14}" for s in SCENARIO_PLAN)
    print(header)
    print("-" * 72)
    for policy in POLICIES:
        line = f"{policy:<20} "
        for sc in SCENARIO_PLAN:
            a = agg2[(policy, sc)]
            line += f"{a['e2e']/a['n']:<14.2f} "
        print(line)
    print("=" * 72)

    # ---- Markdown ----
    md = "results/tables/language_results.md"
    with open(md, "w") as f:
        f.write("# Language-Grounded Navigation Results\n\n")
        f.write("## Overall by policy\n\n")
        f.write("| Policy | N | Parse | Grounding | Navigation | End-to-End | Verified |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for policy in POLICIES:
            a = agg[policy]
            f.write(f"| {policy} | {a['n']} | "
                    f"{a['parse']/a['n']:.2f} | {a['ground']/a['n']:.2f} | "
                    f"{a['nav']/a['n']:.2f} | {a['e2e']/a['n']:.2f} | "
                    f"{a['verif']/a['n']:.2f} |\n")

        f.write("\n## End-to-end by (policy, scenario)\n\n")
        f.write("| Policy | " + " | ".join(SCENARIO_PLAN) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(SCENARIO_PLAN)) + "|\n")
        for policy in POLICIES:
            row = f"| {policy} | "
            row += " | ".join(
                f"{agg2[(policy, sc)]['e2e']/agg2[(policy, sc)]['n']:.2f}"
                for sc in SCENARIO_PLAN
            )
            row += " |\n"
            f.write(row)

    print(f"Saved: {md}")


if __name__ == "__main__":
    main()