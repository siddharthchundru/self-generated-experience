# Stage 1 evaluation and Mac compute estimate

This is harness validation, not a learning experiment. The learner remains Stage 2.
The classifier/M0 dependency is `stage1-route-classifier` at `27881da`, combined
locally with main's public v0.9 protocol at `67236e6`.

## Human classifier check

The author labelled all 60 trajectories independently before the answer key was
computed. There are 56 certain labels with 56/56 agreement (21 north, 24 south,
11 neither). IDs 13, 35, 48 and 59 were marked uncertain; all four also agree,
but are excluded from the certain agreement figure. This is a deliberately
selected challenge sheet, not an unbiased estimate of deployment accuracy.

The answer line, all comparisons, and SHA-256 fingerprints of the displayed
trajectory data, classifier and maze definitions are in
`results/stage1/route_handcheck/human_comparison.json`. The marker and clearer
instructions were added during the check; data and IDs were unchanged. A crossing
is movement across the marked midline in either direction; touching and returning
is not a crossing. The last crossing determines the label, even on failed episodes.

To reproduce, save the answer line to a text file and run:

```sh
python scripts/check_route_labels.py --answers-file /path/to/answer-line.txt
```

## Concrete evaluation protocol

`src/eval/` evaluates M1 and M0 ring tasks. It does not apply a two-route rule to
M2. All same-row west-east pairs are evaluated; the tie row is reported separately
and is excluded from primary aggregates.

- P1 blocks the shorter arm's middle cell: **M1 north (1,4), south (7,4)**.
  For M0 these are (1,5) and (5,5). The choice uses geometry, never outcomes.
- Default Stage 1 T is 100, sampled n is 100, minimum successes is 20, and
  temperature is 1. There are 20 greedy attempts per pair. Best-of-m uses m=5
  and 20 disjoint groups of those 100 blocked sampled attempts. These are
  recorded planning defaults; m, greedy count and the final T are not registered.
- Seeds are deterministic and separated by phase, row and attempt. They do not
  depend on policy name, so reset noise is paired across policies.
- Success is final-step success. First goal arrival, reward sum, final distance
  and path length are recorded separately. Fixed horizons and unmoving goals
  are asserted during every episode.
- Rep is min(north successes, south successes) / total successes. Under 20
  successes it is null, not zero. Reports include exclusions, zero-scored
  sensitivity, a rollout-bootstrap 95% interval and minority successes / all attempts.
- `common_pair_rep` compares conditions on the common qualifying rows only.
- Per-route quality uses successful attempts / all attempts with that last-crossing
  route. Time to goal is averaged over final-success attempts in the route,
  in both actions and seconds. Neither-route failures remain in total-rollout counts.
- Route counts at 5%, 10% and 20% and conditional route entropy are reported per pair.
  No-success route counts are 0; no-success entropy is null.
- A_0 and best-of-m use sampled attempts. S_in and post-adaptation A_k use greedy
  attempts. A successful 'neither' trajectory is an error, never dropped.
- P1-own uses all greedy crossing labels to choose the most-used route. Ties
  choose the shorter route; no crossing makes P1-own undefined for that pair.
  A_k is stratified by which route is blocked.

The expert stand-in chooses the shorter route greedily and chooses north/south
with equal probability per sampled episode. It knows base geometry but is **not
told the evaluation wall**. Thus it cannot plan around the new block by oracle.
The random stand-in has no greedy action: its usual uniform random action rule
also fills the competence/post-adaptation slots, explicitly flagged in reports.

## Adaptation boundary

There is one TOTAL k-step adaptation budget per policy and perturbation, across
the eligible blocked task pairs. Each perturbation starts from a separate deep
copy of the original policy. There is no sixfold multiplication of k by task pair.

The adapter contract receives all blocked pairs and the total budget. It must
start with an empty data buffer, collect C3-style practice, return a policy and
report environment and gradient steps. Stage 2 must implement the actual learner
and settle optimiser-state handling and the update ratio.

`no_update_adaptation` collects exactly k interactions and performs **zero weight
updates**. Its output is named `standin_no_parameter_updates` everywhere. Its A_k
is a control measurement, not learned recovery. Without an adapter, A_k is null.
Stage 1 exercises this interface; it cannot establish the future learner's
budget parity or adaptation effectiveness.

## Run

```sh
pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/evaluate_stage1.py
python scripts/evaluate_stage1.py --maze M0 --quick   # writes results/stage1/evaluation-m0-smoke.json
```

`--quick` lowers sample counts for a smoke check and is not the full gate.
The default output name follows `--maze` and `--quick`, and the script refuses
to write into `results/` from a tree with uncommitted changes unless given
`--allow-dirty`. The committed Stage 1 results were produced with episodes of
100 steps; since Stage 2 every episode is T = 60 steps (`src/episode.py`), so a
re-run will not reproduce them exactly.
Reports store per-attempt records, per-pair denominators, aggregate outcomes,
configuration, package versions and source fingerprints. S_out, P2/P3, BMD's
mode-weighted metric, training exposure and the Stage 2 analysis are not silently
substituted with approximations; they are outside this ring-task implementation.

## Mac benchmark and total compute

```sh
pip install -r requirements-benchmark.txt
python scripts/benchmark_mac.py
python scripts/estimate_compute.py
```

Run the benchmark without competing tests or evaluation. It measures random and
expert simulation, sampled batch-one inference, a synthetic weighted-classification
Adam update on the specified 6-256-256-81 network, and a NumPy KDE scoring surrogate.
The network benchmark is CPU float32, one thread, batch 256. It is not a learner
implementation. Real replay sampling, logging and selector integration are absent.

`mac_throughput.json` records repeat timings, machine information, versions and
source fingerprints. `compute_estimate.json` counts:

- Every weighted main condition and both plain arms; C0 has zero collection steps
  but the same practice-update count.
- Shared initial weighted/plain pretraining and D0 generation.
- A fresh re-learning run for every weighted condition, including C0.
- Two k-step adaptations per main policy, P1 and P1-own, and their evaluations.
- Six-seed pilot, additional gamma sensitivity and the C0 k-selection grid.
- Equal M0 C3/C4 tuning and weighted/plain capacity checks on M1/M2.
- Initial/final evaluation, inference and frontier selector calls.

All unchosen settings are explicit scenario assumptions. Defaults include T=100,
one update per interaction, 10,000 warm-up updates, 140 D0 episodes, k=10% of N,
six tuning trials per condition at three seeds and N=100,000, and optional C1
included. JSON overrides can disable C1, add pilot re-learning or change assumptions.
The scenarios use N=100,000 / 1,000,000 / 5,000,000 and 6 / 20 main seeds.

Totals assume serial CPU work and include 25% and 100% overhead sensitivity cases;
these are **not confidence intervals**. Do not freeze the seed cap solely from
synthetic throughput. Re-measure with the actual learner/replay sampler and final
T/update ratio before committing to main-run compute. Stage 3, development reruns
and manipulation experiments are excluded explicitly.

## Gate status

Human classifier evidence is present, and the evaluation pipeline can run end to
end on stand-ins. The repository's required **independent audit is still pending**;
an implementer's own tests are not that audit. The Stage 1 gate remains open until
that review, the chosen review/PR process, and any findings are handled.
