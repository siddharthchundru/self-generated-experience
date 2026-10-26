# Third audit: commits 2b80a0a and 16cae5c, plus citation check (2026-10-26)

**How:** independent agent; read both diffs; installed requirements.txt with pip (Python **3.13.17**, matching the stated version; install succeeded); ran `python -m unittest discover -s tests` (29 tests, OK); ran mutations in a scratch copy; checked every arXiv/DOI link against the arXiv API and Crossref. Nothing in either repo was modified.

**Verdict.** All five claimed fixes are implemented. Four are tested and caught by mutation; `render_fps` (#4) is set but not tested. No contradiction with docs/spec.md. All 33 arXiv/DOI links resolve to the paper named, with matching first author and year; no misattributions.

## Second-audit findings re-checked

| # | Claimed fix | Implemented | Tested / mutation caught |
|---|---|---|---|
| 1 | None-goal guard | Yes: `options.get(...) is None` raises for start or goal | Yes (None goal mutation fails 1 test) |
| 2 | `allowed_goals` defaults to R_d on M1; M2 = all free | Yes (`DEFAULT_ALLOWED_GOALS`); asserts allowed ⊆ free | Yes (default→all free, and check removed, both fail) |
| 3 | Speed cap before every substep | Yes: `point.step` replaced; `_clip_velocity()` then `do_simulation(action,1)` ×10 | Yes: equivalence vs ten library steps at full force for 1 s, plus a speed-≤5.5 test |
| 4 | `render_fps` = 10 | Yes, on inner and point env metadata | **No test** (mutation to 50 passes) |
| 5 | Layout pins for M1/M2 | M2 == library `MEDIUM_MAZE`; M1 held-out cells listed by hand + `len(R_d)==24` | Yes (flip, single-cell edits fail) |

## Findings

| # | Severity | Finding | Suggested disposition |
|---|---|---|---|
| 1 | low | `render_fps` has no test. | Add a one-line assertion, or accept (only affects rendering). |
| 2 | low | `step` is monkeypatched on the instance and re-implements the library's `PointEnv.step` body (clip action, `_clip_velocity`, `do_simulation`, `_get_obs`). A library upgrade could diverge silently except where the equivalence test catches it. | Accept while versions are pinned; the equivalence test is the right guard. Note in docstring (already partly noted). |
| 3 | low | M1 pin is partial: it pins R_h exactly and |R_d|=24, but not R_d's cells. A change that moves a ring cell while keeping count 24 would rely on the structural/route tests. Row reversals and the edits tried were all caught. | Optional: pin the full M1 grid literally. |
| 4 | info | Guard checks start only for "free", so starts in R_h are allowed. Consistent with spec (P3 uses held-out starts; leakage rule is about goals). Hindsight relabelling in GCSL can still create R_h goals from achieved states; the guard cannot see that. | Carry to the data-collection/learner audit: assert relabelled goals for training respect the leakage rule (spec line 91). |
| 5 | info | `env.unwrapped.reset()` bypass remains; documented. | As already disposed. |
| 6 | info | Spec consistency: 10 Hz (`dt` asserted 0.1), `SUCCESS_RADIUS=0.45`, `POSITION_NOISE=0.25`, 100 steps unchanged; R_d-only training goals now enforced, matching spec line 91. | — |
| 7 | info | Commit 2b80a0a bundles two ideas (speed cap and goal guard); message justifies it. Tests pass at HEAD. | Accept. |

## Mutations (scratch copy)

| Mutation | Caught |
|---|---|
| Per-substep cap removed | Yes (2 failures) |
| Cap applied once per 10 Hz step (old behaviour) | Yes (2 failures) |
| None goal allowed | Yes |
| Goal-in-allowed-set check disabled | Yes (2 failures) |
| Default `allowed_goals` = all free cells | Yes |
| Start-free check disabled | Yes |
| `render_fps` 50 | **No** |
| M1 flipped top to bottom | Yes |
| M1 row 1 reversed | Yes |
| M1 held-out cell (6,11) walled | Yes |
| M1 held-out cell (7,9) walled | Yes |
| M2 free cell (3,4) walled | Yes |

## Citations

All checked against the arXiv API (title, authors, first submission year) and Crossref for DOIs.

| Link | Cited as | Actual title / first author / year | OK? |
|---|---|---|---|
| 2609.36720 | Liu et al., 2026; T²Mem (fast-weight memory) | T²Mem: Learning Test-Time Memory for Robotics / Yize Liu / 2026 | OK |
| 2610.20318 | Alfarra et al., 2026; TEMPO (gradient-based critic-driven) | Efficient Test-Time Adaptation For Robot Policies / Motasem Alfarra / 2026; abstract names method TEMPO, critic-driven | OK |
| 2610.12185 | Chen et al., 2026; RESETTLE (retrieval) | RESETTLE: Robotic Recovery through Disagreement-Triggered Retrieval… / Yuxin Chen / 2026 | OK |
| 2605.10094 | Zhao et al., 2026; Retrieve-then-Steer | Retrieve-then-Steer: Online Success Memory for TTA of Generative VLAs / Jianchao Zhao / 2026 | OK |
| 2605.11387 | Longhini et al., 2026 (mode collapse; BMD) | Behavioral Mode Discovery for Fine-tuning Multimodal Generative Policies / Alberta Longhini / 2026 | OK |
| 2509.15155 | Ghasemipour et al., 2025; Self-Improving EFMs | Self-Improving Embodied Foundation Models / Seyed Kamyar Seyed Ghasemipour / 2025 | OK |
| 2407.20635 | Zhou et al., 2024; SOAR | Autonomous Improvement of Instruction Following Skills via Foundation Models / Zhiyuan Zhou / 2024 (system is SOAR; name not in abstract) | OK |
| 2609.30558 | Ding & Wu, 2026 (MemProbe) | Probing Stability-Plasticity Tradeoffs in Agent Memory… / Jiaqi Ding, Guorong Wu / 2026; abstract names MemProbe | OK |
| 2604.04182 | **Not cited anywhere** in either repo's README/spec/notes | Comparative reversal learning reveals rigid adaptation in LLMs under non-stationary uncertainty / Haomiaomiao Wang / 2026 | n/a — see note |
| 2406.15917 | Du et al., 2024 | To Err is Robotic… / Maximilian Du / 2024 | OK |
| 2509.25424 | Ibn Hamid et al., 2025 | Polychromic Objectives for RL / Jubayer Ibn Hamid / 2025 | OK |
| 1912.06088 | GCSL | Learning to Reach Goals via Iterated Supervised Learning / Dibya Ghosh / 2019 | OK |
| 2007.02832 | MEGA (ICML 2020) | Maximum Entropy Gain Exploration… / Silviu Pitis / 2020 | OK |
| 2110.14020 | Ostrovski et al., 2021 | The Difficulty of Passive Learning in Deep RL / Georg Ostrovski / 2021 | OK |
| 2201.13425 | ExORL | Don't Change the Algorithm, Change the Data / Denis Yarats / 2022 | OK |
| 2010.14484 | SMERL (NeurIPS 2020) | One Solution is Not All You Need… / Saurabh Kumar / 2020 | OK |
| 2505.01396 | Jin et al., 2025a | SIME / Yang Jin / 2025 | OK |
| 2509.19292 | Jin et al., 2025b | SOE / Yang Jin / 2025 | OK |
| 2505.19789 | Liu et al., 2025 | What Can RL Bring to VLA Generalization? / Jijia Liu / 2025 | OK |
| 2504.13837 | Yue et al., 2025 | Does RL Really Incentivize Reasoning Capacity…? / Yang Yue / 2025 | OK |
| 2306.05727 | Weltevrede et al., 2023 | The Role of Diverse Replay for Generalisation in RL / Max Weltevrede / 2023 | OK |
| 2410.03565 | Weltevrede et al., 2024 | Training on Irrelevant States Implies Data Augmentation / Max Weltevrede / 2024 | OK |
| 2603.25464 | Hu et al., 2026 | Maximum Entropy Behavior Exploration for Sim2Real Zero-Shot RL / Jiajun Hu / 2026 | OK |
| 2411.01521 | Lintunen et al., 2024 | Diversity Progress for Goal Selection… / Erik M. Lintunen / 2024 | OK |
| 2502.07709 | MAGELLAN | MAGELLAN: Metacognitive predictions of learning progress… / Loris Gaven / 2025 | OK |
| 1606.04460 | Blundell et al., 2016 | Model-Free Episodic Control / Charles Blundell / 2016 | OK |
| 1703.01988 | Pritzel et al., 2017 | Neural Episodic Control / Alexander Pritzel / 2017 | OK |
| 2211.15183 | Yang et al., 2022 | Continuous Episodic Control / Zhao Yang / 2022 | OK |
| 2410.02995 | Yang et al., 2024 | Task-agnostic Lifelong Robot Learning with Retrieval-based Weighted Local Adaptation / Pengzhi Yang / 2024 | OK |
| 2606.15631 | Park et al., 2026 | Retrieve, Don't Retrain… / Jeongeun Park / 2026 | OK |
| 2503.15386 | Razmjoo et al., 2025 | CCDP: Composition of Conditional Diffusion Policies… / Amirreza Razmjoo / 2025 | OK |
| 2609.18167 | Yang, Thompson & Konidaris, 2026 | Characterizing Replay Retention Under Dynamics Shift in MBRL / Everest Yang, Skye Thompson, George Konidaris / 2026 | OK |
| doi 10.3389/fnhum.2014.00590 | Egbert & Barandiaran, 2014 | Modeling habits as self-sustaining patterns of sensorimotor behavior / Egbert / 2014 | OK |
| doi 10.3389/fnbot.2022.846693 | Woolford & Egbert, 2022 | Goal Oriented Behavior With a Habit-Based Adaptive Sensorimotor Map Network / Woolford / 2022 | OK |

**Note on 2604.04182.** It is not linked in either repo. It is a reversal-learning paradigm applied to LLMs (behavioural diagnostic, non-stationary). switch-or-recalibrate's literature gate (line 40) says the search of "reversal-learning tests" found nothing that fails the gate. This paper does not fail the gate (LLMs, not sensorimotor TTA mechanisms), but it is a direct precedent for "reversal learning as a diagnostic of adaptation", alongside MemProbe. Suggest citing it as precedent in the gate's table. The non-arXiv RoboTTT link (research.nvidia.com) was not checked.

Claims checked where obvious: TEMPO is critic-driven (abstract confirms); T²Mem is test-time memory; Retrieve-then-Steer and RESETTLE are retrieval-based; BMD concerns modes in fine-tuning multimodal policies. Papers were checked from metadata/abstracts only.
