"""Measure Mac simulation, inference, synthetic updates and frontier-selector cost.

Run alone (not beside tests or evaluation): python scripts/benchmark_mac.py
PyTorch is benchmark-only, not an implementation of the Stage 2 learner.
The synthetic weighted classification update uses the specified architecture;
it excludes a real replay sampler, logging and framework integration overhead.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from envs.mazes import M1
from envs.pointmaze import make_env
from episode import EPISODE_STEPS
from expert import run_episode


def timed(fn, count):
    start = time.perf_counter()
    for _ in range(count):
        fn()
    return time.perf_counter()-start


def describe(times, operations):
    rates = [operations/t for t in times]
    return {"repeats": len(times), "operations_per_repeat": operations,
            "elapsed_seconds": times, "rates_per_second": rates,
            "median_per_second": statistics.median(rates),
            "slowest_per_second": min(rates)}


def sysctl(key):
    result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def benchmark_simulator(repeats, count, horizon):
    env = make_env("M1", max_episode_steps=horizon)
    rng = np.random.default_rng(0)
    index = 0
    def reset():
        env.reset(seed=0, options={"reset_cell": [4, 1], "goal_cell": [4, 7]})
    def step():
        nonlocal index
        _, _, _, truncated, _ = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
        index += 1
        if truncated:
            reset()
    reset()
    timed(step, 1000)
    random_times = [timed(step, count) for _ in range(repeats)]
    episodes = max(1, count//horizon)
    def expert():
        run_episode(env, M1, (2, 1), (2, 7), "south", seed=0)
    timed(expert, 5)
    expert_times = [timed(expert, episodes) for _ in range(repeats)]
    env.close()
    return {"random_actions": describe(random_times, count),
            "expert_episodes": describe(expert_times, episodes*horizon), "horizon": horizon}


def benchmark_network(repeats, updates, batch_size, horizon):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 81))
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, 6)
    targets = torch.randint(81, (batch_size,))
    weights = 0.96205 ** torch.randint(1, horizon+1, (batch_size,)).float()
    def update():
        optimiser.zero_grad(set_to_none=True)
        losses = nn.functional.cross_entropy(model(x), targets, reduction="none")
        loss = (weights*losses).sum()/weights.sum()
        loss.backward()
        optimiser.step()
    timed(update, 50)
    update_times = [timed(update, updates) for _ in range(repeats)]
    single = x[:1]
    def inference():
        with torch.no_grad():
            probabilities = model(single).softmax(-1)
            torch.multinomial(probabilities, 1).item()
    timed(inference, 200)
    inference_times = [timed(inference, 2000) for _ in range(repeats)]
    return {"device": "cpu", "threads": 1, "dtype": "float32", "batch_size": batch_size,
            "architecture": [6, 256, 256, 81], "parameters": sum(p.numel() for p in model.parameters()),
            "optimizer": "Adam", "synthetic_updates": describe(update_times, updates),
            "sampled_batch1_inference": describe(inference_times, 2000),
            "torch_version": torch.__version__,
            "limitation": "Fixed synthetic batches; actual replay sampling and logging not measured. CPU only; no MPS claim."}


def benchmark_selector(repeats):
    # Representative Gaussian KDE scoring, not the unimplemented C4 selector.
    rng = np.random.default_rng(0)
    reference = rng.uniform(-1, 1, (10000, 2))
    candidates = rng.uniform(-1, 1, (100, 2))
    def score():
        distance_sq = ((candidates[:, None, :]-reference[None, :, :])**2).sum(axis=-1)
        return np.exp(-distance_sq/(2*0.1**2)).mean(axis=1).argmin()
    timed(score, 3)
    times = [timed(score, 20) for _ in range(repeats)]
    return {"implementation": "NumPy Gaussian kernel scoring surrogate",
            "reference_points": 10000, "candidates": 100, "bandwidth": 0.1,
            "selections": describe(times, 20),
            "limitation": "Includes distance/kernel scoring, excludes real buffer sampling and selector integration."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--env-steps", type=int, default=20000)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=EPISODE_STEPS)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stage1/mac_throughput.json")
    args = parser.parse_args()
    result = {
        "machine": {"system": platform.system(), "release": platform.release(),
                    "architecture": platform.machine(), "python": platform.python_version(),
                    "chip": sysctl("machdep.cpu.brand_string"), "memory_bytes": sysctl("hw.memsize"),
                    "physical_cores": sysctl("hw.physicalcpu"), "logical_cores": sysctl("hw.logicalcpu")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in (ROOT/"src", ROOT/"scripts") for p in sorted(folder.rglob("*.py"))},
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "gymnasium", "gymnasium-robotics", "mujoco")},
        "simulator": benchmark_simulator(args.repeats, args.env_steps, args.horizon),
    }
    print("Simulator measurements complete", flush=True)
    result["network"] = benchmark_network(args.repeats, args.updates, args.batch_size, args.horizon)
    print("Network measurements complete", flush=True)
    result["selector_surrogate"] = benchmark_selector(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"random_env_sps": result["simulator"]["random_actions"]["median_per_second"],
                      "expert_env_sps": result["simulator"]["expert_episodes"]["median_per_second"],
                      "updates_per_second": result["network"]["synthetic_updates"]["median_per_second"],
                      "sampled_inferences_per_second": result["network"]["sampled_batch1_inference"]["median_per_second"]}, indent=2))


if __name__ == "__main__":
    main()
