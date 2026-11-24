"""The re-learning control: a fresh learner trained on a finished run's final buffer.

What this file does, in one sentence: `run_relearning` takes one finished
training run (its settings, summary and final buffer), trains a new learner on
that buffer with exactly the run's gradient-step count, and returns the new
learner's policies and logs, without touching an environment.

Why it matters (spec section 7, H4; Ostrovski et al., 2021): in the original
run, data arrived over time and the policy that was learning also chose what to
collect next. The re-learner gets the same final data all at once and never
collects. If a condition's advantage reproduces, the final dataset is enough to
produce it; if it does not, the advantage depends on how and when the data
arrived. Section 9 states the limits of both readings.

How it is done (decision 13 of the 4 Nov audit, Claude's recommendation, still
open; the defaults below follow it):
- Phases in the original order. Warm-up on D0 alone, then an Adam reset (spec
  v0.12), then N x r updates on the whole final buffer, where r is the run's
  updates per environment step. The total is warm-up plus N x r, the original
  run's total, and it is checked against the run's own budget audit.
- Warm-up length. The re-learner takes exactly as many warm-up updates as the
  original run recorded, with the fixed rule. If the original used the
  validation rule, re-running that rule could stop at another length and break
  the match, so the recorded length is reused (Claude's choice, for review).
- Initial weights, `init` (open):
  - "fresh" (default, decision 13): new initial weights and a new warm-up batch
    stream, both derived from the seed with tags of their own. The re-learner is
    a different network from the start, as in Ostrovski et al.'s tandem setup.
  - "same": the original's initial weights and warm-up stream, so warm-up
    reproduces the original pretrained policy exactly (checked against its
    fingerprint). Only the arrival of the practice data then differs.
- Practice batches come from their own random stream. Every example draws a
  trajectory uniformly from the final buffer (GCSL's procedure, unchanged), so
  D0's share is fixed at its final value from the first update, where in the
  original run it fell as data accumulated. That difference is the point of the
  control, not a side effect.
- The learning-rate fallback of spec v0.12 (`lr_warmup_steps`) applies at the
  reset exactly as in the original run.
- Checks: the buffer's digest matches the run's summary; the run passed its own
  budget audit; D0 is the buffer's first trajectories and has the recorded
  count; every optimiser step in the process is counted, and the total must
  equal the original's warm-up plus practice updates.
"""

from dataclasses import asdict, replace
import time

import numpy as np
from torch.optim.optimizer import register_optimizer_step_post_hook

from gcsl import GCSLLearner, TrajectoryBuffer, updates_per_episode
from training import TAG_INIT, TAG_WARMUP, _adam_steps, _Exposure, _step, warm_up

TAG_RELEARN_INIT = 0xB1
TAG_RELEARN_WARMUP = 0xB2
TAG_RELEARN_PRACTICE = 0xB3
INITS = ("fresh", "same")


def check_source(settings, summary, buffer):
    """Refuse a run that is unfinished, failed its audit, or whose buffer is not the one it saved."""
    if summary.get("kind", "training") != "training":
        raise ValueError("the re-learning control takes a training run, not a re-learning run")
    if summary["settings"] != asdict(settings):
        raise ValueError("the run's settings.json and summary.json disagree")
    budget = summary["budget"]
    if budget.get("passed") is not True:
        raise ValueError("the run did not pass its budget audit")
    if buffer.digest() != summary["final_buffer_digest"]:
        raise ValueError("the saved buffer differs from the one the run recorded")
    if len(buffer) != budget["expected_buffer_trajectories"]:
        raise ValueError("the saved buffer holds a different number of trajectories than the run collected")
    demos = summary["demonstrations"]
    if buffer.sources[:demos] != ["D0"] * demos or "D0" in buffer.sources[demos:]:
        raise ValueError("D0 must be the buffer's first trajectories, and only those")
    if budget["practice_gradient_steps"] != settings.practice_updates:
        raise ValueError("the run's practice updates differ from N x ratio")


def run_relearning(settings, summary, buffer, init="fresh", on_chunk=None):
    """Train a new learner on `buffer`, the final buffer of the run described by
    `settings` and `summary`. See the module docstring.

    `on_chunk(row)` receives one log row per T x ratio updates, matching the
    original run's per-episode rows. Returns a dict with the summary, the chunk
    log and the re-learner's pretrained and final weights.
    """
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}")
    if settings.learner != "weighted" and settings.purpose in ("pilot", "main"):
        raise ValueError("the re-learning control is for the main (weighted) learner only (spec section 7)")
    check_source(settings, summary, buffer)
    started = time.perf_counter()
    counted = [0]
    handle = register_optimizer_step_post_hook(lambda *_: counted.__setitem__(0, counted[0] + 1))
    try:
        return _relearn(settings, summary, buffer, init, on_chunk, started, counted)
    finally:
        handle.remove()


def _relearn(s, original, buffer, init, on_chunk, started, counted):
    demos = original["demonstrations"]
    arrays = buffer.to_arrays()
    d0 = TrajectoryBuffer.from_arrays({k: v[:demos] for k, v in arrays.items()})

    # Warm-up: the original's recorded length, fixed rule.
    warmup_updates = original["budget"]["warmup_gradient_steps"]
    warmup_settings = replace(s, warmup_rule="fixed", warmup_updates=warmup_updates)
    tags = (TAG_INIT, TAG_WARMUP) if init == "same" else (TAG_RELEARN_INIT, TAG_RELEARN_WARMUP)
    policy, warmup_log = warm_up(warmup_settings, d0, init_tag=tags[0], stream_tag=tags[1])
    if counted[0] != warmup_updates or warmup_log["optimizer_steps"] != warmup_updates:
        raise AssertionError("re-learning warm-up took a different number of optimiser steps")
    pretrained_state = {k: v.clone() for k, v in policy.network.state_dict().items()}
    pretrained_digest = policy.parameter_digest()
    if init == "same" and pretrained_digest != original["pretrained_digest"]:
        raise AssertionError("init='same' must reproduce the original pretrained policy exactly")
    if init == "fresh" and pretrained_digest == original["pretrained_digest"]:
        raise AssertionError("init='fresh' reproduced the original pretrained policy")

    # Adam reset (spec v0.12), then N x ratio updates on the whole final buffer.
    learner = GCSLLearner(policy, s.gamma, s.learning_rate, s.batch_size, s.lr_warmup_steps)
    if learner.optimizer.state or policy.parameter_digest() != pretrained_digest:
        raise AssertionError("re-learning practice must start from its warm-up weights with an empty Adam state")
    rng = np.random.default_rng(np.random.SeedSequence([s.seed, TAG_RELEARN_PRACTICE]))
    per_chunk = updates_per_episode(s.updates_per_env_step)
    exposure = _Exposure(s.maze)
    log = []
    for index in range(s.practice_episodes):
        exposure.reset()
        for _ in range(per_chunk):
            _step(learner, buffer, rng, exposure)
        row = {"chunk": index, "gradient_steps": learner.gradient_steps,
               "learning_rate": learner.current_learning_rate(), "batches": exposure.summary()}
        log.append(row)
        if on_chunk is not None:
            on_chunk(row)

    expected_total = original["budget"]["warmup_gradient_steps"] + original["budget"]["practice_gradient_steps"]
    budget = {
        "warmup_gradient_steps": warmup_updates,
        "practice_gradient_steps": learner.gradient_steps,
        "practice_optimizer_steps": _adam_steps(learner.optimizer),
        "expected_practice_gradient_steps": original["budget"]["practice_gradient_steps"],
        "all_optimizer_steps_in_process": counted[0],
        "original_total_gradient_steps": expected_total,
        "environment_steps": 0,
        "buffer_trajectories": len(buffer),
    }
    if not (learner.gradient_steps == budget["practice_optimizer_steps"]
            == budget["expected_practice_gradient_steps"]) or counted[0] != expected_total:
        raise AssertionError(f"re-learning budget audit failed: {budget}")
    budget["passed"] = True
    summary = {
        "kind": "relearning", "init": init,
        "settings": asdict(s), "gamma": s.gamma,
        "source": {"pretrained_digest": original["pretrained_digest"],
                   "final_digest": original["final_digest"],
                   "final_buffer_digest": original["final_buffer_digest"],
                   "warmup_rule": s.warmup_rule},
        "demonstrations": demos,
        "warmup_key": s.warmup_key(),
        "pretrained_digest": pretrained_digest, "final_digest": policy.parameter_digest(),
        "warmup": warmup_log, "budget": budget,
        "source_mix": {source: buffer.sources.count(source) for source in sorted(set(buffer.sources))},
        "wall_clock_seconds": {"total": time.perf_counter() - started},
    }
    return {"summary": summary, "chunks": log, "pretrained_state": pretrained_state,
            "final_state": {k: v.clone() for k, v in policy.network.state_dict().items()}}
