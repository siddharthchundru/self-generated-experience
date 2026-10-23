# Audit: simulator integration (commit c84aa19)

**Date:** 23 October 2026
**Audited:** `src/envs/pointmaze.py`, `tests/test_pointmaze.py`, `requirements.txt`
**How:** a separate agent that had not seen the code being written read the commit and the spec, ran the tests and its own throwaway scripts, and modified nothing. The implementing agent then re-ran the two high-severity checks independently and reproduced both (`frame_skip 1`, `dt 0.01`; 168/500 goals in the held-out region).

**Verdict.** The geometry is correct: MuJoCo's wall geoms match the M1 grid exactly (91 of 91 walls, no flip or transposition), cell/coordinate conversion is exact off-centre, and seeded rollouts are deterministic. The problems are inherited defaults and documentation. Items 1–3 must be resolved before any data-collection or learning code relies on this environment.

## Findings

| # | Severity | Finding | Evidence | Suggested fix | Disposition |
|---|---|---|---|---|---|
| 1 | high | Control rate is 100 Hz (`dt` 0.01, `frame_skip` 1), not the spec's 10 Hz. Step budgets mean different things at each rate. | `point_env.dt`, `point_env.frame_skip`; spec §4 | Set `frame_skip=10`, or correct the spec | Fixed: 10 Hz chosen; `frame_skip` set to 10 after construction (fed5484) |
| 2 | high | Goal-leakage risk: `reset()` without `goal_cell` samples goals anywhere, including the held-out region (168/500). Starts likewise. | 500 unseeded-option resets | Wrapper that requires explicit cells or samples from an allowed set; test for it | Fixed: `RequireCells` wrapper; test added (89bc829, then tests) |
| 3 | medium | `continuing_task=True` is inherited: success never ends the episode and sparse reward 1.0 is paid every step within 0.45 of the goal. | Library defaults; P-controlled rollout to goal | Set `continuing_task`, `reward_type`, `reset_target` explicitly; record the 0.45 threshold in the spec | Fixed: success ends the episode; settings explicit; spec §4 updated (fed5484) |
| 4 | medium | ±0.25 position noise on start and goal even when cells are given; with the 0.45 radius, success can register from a neighbouring cell. | 300 resets, max offset 0.2495 | Make noise explicit; decide whether success and routes use cell or position | Fixed: noise set explicitly to 0.25 and in the spec. Success uses position (library radius); routes use cells (fed5484) |
| 5 | medium | Tests are weak: centre round-trip would pass for any grid; the wall test checks library self-consistency and its random walk never reached e.g. (7,6), (5,7). | Visited-cell list | Geom-vs-grid test; hand-computed anchors; off-centre and boundary cases; scripted push into a known wall | Fixed: tests replaced; a flipped M1 now fails the wall test |
| 6 | low | Cell-boundary convention is half-open and undocumented; points outside the maze return invalid indices without error. | Boundary probes | Document; bounds-check `xy_to_cell` | Fixed: conversion computed from the grid, documented, bounds-checked (6366611) |
| 7 | medium | `requirements.txt` omits `gymnasium` (1.4.0) and leaves `numpy` (2.5.3) unpinned, though the commit message says "pinned"; no Python version stated (3.13). | `pip show` | Pin both; state Python version | Fixed |
| 8 | low | Building on the `PointMaze_UMaze-v3` id means its defaults apply silently. | `_BASE_ID` | Pass every relevant kwarg explicitly | Fixed with 3 |
| 9 | info | Grid fidelity confirmed. | Geom set equals wall set | — | — |
| 10 | info | Determinism confirmed with `reset(seed, options)`. | 100 identical actions, equal observations | Don't rely on unseeded resets in experiments | — |
| 11 | low | `mazes.py` says the corridor leaves the ring "at (4, 7)"; (4, 8) is the first corridor cell. R_h is defined only as "everything east". | M1 comment | Name (4, 8); define R_h as a cell set | Fixed: `M1_DEMO_REGION`, `M1_HELD_OUT_REGION` (89bc829) |
| 12 | info | Action space `Box(-1, 1, (2,))` matches the spec. | — | — | — |

All findings were addressed; a second audit (`2026-10-23-simulator-fixes.md`) found that 1, 2 and 5 were only partly fixed, and those gaps are closed there. The episode-length placeholder also changed from 300 to 100 steps, because 300 steps at 10 Hz is 30 s.
