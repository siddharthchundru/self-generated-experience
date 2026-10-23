# Audit: fixes to the simulator integration (commits fed5484..a7e641b)

**Date:** 23 October 2026
**Audited:** the six commits that responded to `2026-10-23-simulator-integration.md`.
**How:** a separate agent, which had not seen the fixes being written, read the diff, ran the tests at every commit, tried mutations in a throwaway copy of the repo, and modified nothing.

**Verdict.** The fixes were real, and tests passed at every commit. Three problems remained:
- **Speed.** Setting `frame_skip` after construction changed the dynamics, not just the control rate: the library caps speed once per call, so the ball reached 6.9 instead of about 5.
- **Goal guard.** The guard could be bypassed by passing `None`, and it did not restrict goals to R_d.
- **Grid errors.** An error in `mazes.py` itself (for example a flipped M1 or M2) could not fail any test.

## Prior findings

| # | Status after the six commits | Reason |
|---|---|---|
| 1 (10 Hz) | partly | `dt` correct; speed cap and fps metadata were not (findings 3, 4 below) |
| 2 (goal leakage) | partly | `None` bypass; no restriction to R_d (1, 2 below) |
| 3, 4, 6, 7, 8, 11 | fixed | — |
| 5 (weak tests) | partly | Grid errors in `mazes.py` undetectable (5 below) |

## Findings

| # | Severity | Finding | Disposition |
|---|---|---|---|
| 1 | high | `{"goal_cell": None}` passed the guard; the library then samples anywhere (109/300 goals in R_h). | Fixed (2b80a0a): guard requires non-None, free start, allowed goal; tested |
| 2 | medium | Guard did not restrict goals to R_d; `env.unwrapped.reset()` bypasses it. | Fixed (2b80a0a): `allowed_goals` defaults to R_d on M1; S_out must pass R_h explicitly. The `unwrapped` bypass cannot be closed by a wrapper; it is documented, and the data-collection audit will check every logged goal |
| 3 | medium | Speed cap applied once per 10 substeps: 6.9 at 10 Hz against 5.2 at 100 Hz. | Fixed (2b80a0a): cap applied before every substep; one 10 Hz step now equals ten library steps under full force; speed test added |
| 4 | low | `render_fps` metadata stale (50 and 100). | Fixed (2b80a0a) |
| 5 | medium | Wall test compares simulator with its own input; flipped or transposed M1/M2 passed. | Fixed (16cae5c): M2 checked against the library's `MEDIUM_MAZE`; M1's held-out cells listed by hand |
| 6 | low | Vector-env autoreset calls `reset()` without options and raises. | Noted: fails safely. Any vectorised collector must supply cells through its own reset logic |
| 7 | low | Tests rely on library internals. | Accepted with pinned versions; noted in the test docstring |
| 8 | low | Docstrings and the first audit file overclaimed. | Fixed (2b80a0a and this commit) |
| 9 | info | Spec, code and README consistent on 10 Hz, 100 steps, ±0.25 noise, 0.45 radius. | — |
| 10 | info | Each commit one idea; tests pass at each. | — |

## Mutations tried by the auditor

| Mutation | Caught before these fixes |
|---|---|
| `frame_skip` 1; noise 0 or 0.5; `continuing_task=True`; guard disabled; off-by-one or `round` in `xy_to_cell` | Yes |
| M1 or M2 flipped or transposed in `mazes.py`; an M2 free cell made a wall | No; now caught by the layout pins |
| `None` goal through the guard | No; now tested |

Not yet re-audited: commits 2b80a0a and 16cae5c.
