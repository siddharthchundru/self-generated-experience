"""Generate messy M1 trajectories for the hand-check of the route classifier.

Run from the repository root:  python scripts/route_handcheck.py

What it does: the Stage 1 gate requires the route classifier (src/routes.py)
to be checked by hand on at least 50 trajectories. Expert trajectories are too
clean to test it, so this script drives the ball with four kinds of imperfect
controller on the west-east ring pairs:
- noisy expert: the expert's action plus Gaussian noise;
- indecisive: heads for one route, then switches to the other part-way;
- sloppy: the expert's action, replaced by a random one 30-50% of the time;
- random walk: smoothed random forces (often never gets across).
It generates a large pool, then draws 60 for the sheet, deliberately
over-sampling the hard cases: trajectories that visit both arms, and ones that
never get across. The sheet is therefore harder than typical rollouts; the
agreement rate it gives is a conservative figure, not the classifier's
accuracy on learner data.

Outputs (results/stage1/route_handcheck/):
- trajectories.json: id, pair, controller and positions for each trajectory,
  WITHOUT the classifier's label, for blind hand-labelling;
- classifier_labels.json: the classifier's label for each id, kept separate so
  the labeller does not see it;
- sheet.html: a labelling page built from scripts/handcheck_template.html with
  the trajectories embedded (no classifier labels in it).
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from envs.mazes import M1, route_class  # noqa: E402
from envs.pointmaze import make_env, xy_to_cell  # noqa: E402
from expert import ScriptedExpert  # noqa: E402
from routes import trajectory_route  # noqa: E402

T = 60
POOL_PER_KIND = 80
SHEET_SIZE = 60
OUT = ROOT / "results" / "stage1" / "route_handcheck"


def rollout(env, rng, row, kind):
    start, goal = (row, 1), (row, 7)
    obs, _ = env.reset(seed=int(rng.integers(1 << 30)),
                       options={"reset_cell": list(start), "goal_cell": list(goal)})
    first = str(rng.choice(["north", "south"]))
    other = "south" if first == "north" else "north"
    expert = ScriptedExpert(M1, start, goal, obs["desired_goal"], first)
    switch_at = int(rng.integers(3, 20))
    sigma = float(rng.choice([0.6, 1.0, 1.5]))
    p_random = float(rng.uniform(0.3, 0.5))
    smooth = np.zeros(2)
    positions = [obs["observation"][:2].tolist()]
    for t in range(T):
        o = obs["observation"]
        if kind == "indecisive" and t == switch_at:
            here = xy_to_cell(M1, o[:2])
            expert = ScriptedExpert(M1, here, goal, obs["desired_goal"], other)
        a = expert.act(o)
        if kind == "noisy":
            a = a + rng.normal(0, sigma, 2)
        elif kind == "sloppy" and rng.random() < p_random:
            a = rng.uniform(-1, 1, 2)
        elif kind == "random":
            smooth = 0.8 * smooth + 0.2 * rng.uniform(-1, 1, 2) * 3
            a = smooth
        obs, _, _, _, info = env.step(np.clip(a, -1, 1).astype(np.float32))
        positions.append(obs["observation"][:2].tolist())
    return {"pair": [list(start), list(goal)], "controller": kind,
            "success": bool(info["success"]), "positions": positions}


def main():
    rng = np.random.default_rng(20261028)
    env = make_env("M1", max_episode_steps=T)
    pool = []
    for kind in ("noisy", "indecisive", "sloppy", "random"):
        for _ in range(POOL_PER_KIND):
            row = int(rng.integers(1, 8))
            try:
                ep = rollout(env, rng, row, kind)
            except ValueError:
                continue  # an indecisive switch from a cell with no route of that kind
            pos = np.array(ep["positions"])
            ep["label"] = trajectory_route(pos)
            ep["cell_rule"] = route_class([xy_to_cell(M1, p) for p in pos])
            pool.append(ep)
    env.close()

    hard = [e for e in pool if e["cell_rule"] == "both"]
    neither = [e for e in pool if e["label"] == "neither"]
    plain = [e for e in pool if e not in hard and e not in neither]
    pick = lambda xs, n: [xs[i] for i in rng.permutation(len(xs))[:n]]
    sheet = pick(hard, 25) + pick(neither, 10)
    sheet += pick(plain, SHEET_SIZE - len(sheet))
    sheet = [sheet[i] for i in rng.permutation(len(sheet))]

    OUT.mkdir(parents=True, exist_ok=True)
    blind, labels = [], {}
    for i, e in enumerate(sheet, 1):
        blind.append({"id": i, "pair": e["pair"], "controller": e["controller"],
                      "positions": [[round(x, 3) for x in p] for p in e["positions"]]})
        labels[i] = {"classifier": e["label"], "cell_rule": e["cell_rule"],
                     "success": e["success"], "controller": e["controller"]}
    (OUT / "trajectories.json").write_text(json.dumps(blind))
    (OUT / "classifier_labels.json").write_text(json.dumps(labels, indent=1) + "\n")
    page = (ROOT / "scripts" / "handcheck_template.html").read_text()
    page = page.replace("__GRID__", json.dumps(M1)).replace("__DATA__", json.dumps(blind))
    (OUT / "sheet.html").write_text(page)
    counts = {k: sum(1 for v in labels.values() if v["classifier"] == k)
              for k in ("north", "south", "neither")}
    print(f"pool {len(pool)}; sheet {len(sheet)}; visits both arms {sum(v['cell_rule'] == 'both' for v in labels.values())}; "
          f"classifier labels {counts}")


if __name__ == "__main__":
    main()
