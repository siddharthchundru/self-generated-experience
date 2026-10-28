# Stage 0 literature check

*Siddharth Chundru · 22 October 2026*

**Question for the gate.** Has anyone already run this comparison: one fixed goal-conditioned learner, starting from narrow demonstrations, with conditions that differ only in how goals are chosen during practice (task goals vs. frontier novelty vs. random), measuring competence, policy-level repertoire, and recovery after a route is blocked, plus a re-learning control that separates the data from the active loop?

**Verdict: pass, with a reframing.** I found no paper that runs this comparison. But one half of the central hypothesis — that task-directed improvement collapses behavioural modes — is now well documented in neighbouring settings, so it should be cited as motivation, not claimed as a finding. The contribution moves to the other half: whether *intrinsic goal selection*, with no explicit diversity objective, preserves repertoire, and whether preserved repertoire actually buys adaptation.

## What exists

| Work | What it shows | Overlap | Consequence |
| --- | --- | --- | --- |
| [Longhini et al., 2026](https://arxiv.org/abs/2605.11387) (BMD, ICML 2026) | RL fine-tuning of pre-trained generative policies (e.g. diffusion) collapses demonstrated modes into one reward-maximising mode. An intrinsic mutual-information reward over a discrete latent preserves them. Tested on a 2D four-mode navigation task, ManiSkill, ANYmal and Franka Kitchen; reports mode-weighted success, mode coverage and mode entropy. Perturbation tests (one locked joint, observation noise) on ANYmal only. | **Highest.** Mode collapse under task-directed improvement, measured with mode metrics, in a 2D navigation task. | Cite as the closest work. "Self-practice narrows repertoire" is no longer a headline. Adopt their mode-weighted success alongside route repertoire for comparability. |
| [Ibn Hamid et al., 2025](https://arxiv.org/abs/2509.25424) (Polychromic Objectives) | RL fine-tuning of pretrained policies collapses them onto a few easily exploited outputs; an objective that explicitly rewards diverse generations prevents this and generalises better under large perturbations. BabyAI, MiniGrid, Algorithmic Creativity. | **Medium–high.** Collapse, diversity and perturbation tests in grid mazes. | Diversity comes from an explicit objective with fixed goals, not from goal selection; no re-learning control. Cite with BMD. |
| [Du et al., 2024](https://arxiv.org/abs/2406.15917) ("To Err is Robotic") | At deployment, a demonstration-trained diffusion policy detects stalled progress and resamples among its own candidate actions, away from recorded failures, with no gradient updates. | **Medium–high, as motivation.** | Deployment-time recovery of this kind assumes the base policy still has alternative modes. This project asks which ways of practising keep them. |
| [Jin et al., 2025a](https://arxiv.org/abs/2505.01396) (SIME); [Jin et al., 2025b](https://arxiv.org/abs/2509.19292) (SOE) | In robot self-improvement from demonstrations, policies repeat existing behaviour; injecting structured exploration improves self-improvement. | High, as motivation. | Method papers in manipulation; no controlled comparison of practice strategies, no adaptation tests. Already cited. |
| [Liu et al., 2025](https://arxiv.org/abs/2505.19789) | Fine-tuning OpenVLA with RL (PPO) instead of SFT generalises better on execution shifts. RL rollouts cover more of the workspace, while SFT rollouts cluster on the demonstrations' planner paths (a qualitative observation). | Medium. | In tension with BMD: RL can widen *state coverage* while collapsing *action modes*. This supports measuring coverage on the data and repertoire on the policy, and expecting them to disagree. |
| [Yue et al., 2025](https://arxiv.org/abs/2504.13837) (NeurIPS 2025); [Wu et al., 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/a61aa495e3d56cf793ca4139d45bd877-Paper-Conference.pdf) (ICLR 2026) | In LLMs, RL with verifiable rewards raises pass@1 but narrows the set of problems solvable at large k (Yue et al.). Fine-tuning collapses reasoning modes and limits test-time scaling (Wu et al. show it for SFT and distillation, and cite Yue et al. for RL). | Different domain, same phenomenon. | Framing only: the maze is a transparent instance of a collapse also seen in language-model post-training. |
| [Weltevrede et al., 2023](https://arxiv.org/abs/2306.05727); [Weltevrede et al., 2024](https://arxiv.org/abs/2410.03565) | The diversity of replayed transitions, not just their quantity, drives zero-shot generalisation (2023, DQN). Training on extra explored states acts as data augmentation whose benefit shrinks when value targets on those states are inaccurate (2024, several algorithms). | Medium, for the generalisation outcome. | Confirms that held-out-goal success should stay a secondary sanity check, not a finding. The value-target caveat bites less here: GCSL has no value function, which is worth stating. |
| [Hu et al., 2026](https://arxiv.org/abs/2603.25464) (FB-MEBE) | Within one online Forward-Backward framework trained from scratch, sampling exploration behaviours inversely to their replay-buffer density raises achieved-behaviour entropy and zero-shot return on a quadruped; no perturbation-recovery tests. | Medium. | Independent support that density-inverse (frontier-style) selection preserves diversity. Different embodiment, no demonstrations. |
| [Lintunen et al., 2024](https://arxiv.org/abs/2411.01521) (Diversity Progress) | A learned goal-selection curriculum for discriminability-motivated agents that avoids collapse of the goal distribution. | Low–medium. | Another goal-selection rule; candidate for a later condition, not the MVE. |
| Already in the design: [ExORL](https://arxiv.org/abs/2201.13425), [MEGA](https://arxiv.org/abs/2007.02832) (ICML 2020), [SMERL](https://arxiv.org/abs/2010.14484) (NeurIPS 2020), [Ostrovski et al., 2021](https://arxiv.org/abs/2110.14020) (NeurIPS 2021), [Self-Improving EFMs](https://arxiv.org/abs/2509.15155), [SOAR](https://arxiv.org/abs/2407.20635) | Data-source comparisons, frontier goal selection, diversity-for-adaptation, passive vs. active learning, foundation-model self-practice. | As in the design doc. | No change. |

## What is left (the defensible gap)

1. **Implicit vs. explicit diversity.** BMD and Polychromic Objectives both preserve diversity with an *explicit* objective. No paper I found asks whether *choosing which goals to practise* — frontier novelty, with the learner and its objective unchanged — preserves modes on its own.
2. **Repertoire → adaptation, tested causally.** BMD's perturbation tests are one locked joint and observation noise, on ANYmal only. SMERL shows diversity aids adaptation but trains for diversity explicitly. A blocked-route test on each agent's own dominant route, followed by a common adaptation procedure, is not in this set.
3. **Data vs. loop.** None of the self-improvement papers retrain a fresh learner on the final data to locate where an advantage lives.

## Implications for this project

**The project proceeds, with this framing:** Task-directed improvement is known to collapse modes. Does intrinsically chosen practice preserve them without an explicit diversity objective — and does the preserved repertoire buy adaptation?

Concrete changes:

- README: cite BMD and Polychromic Objectives as the closest work; restate the contribution as above.
- Metrics: add mode-weighted success rate (BMD's SRM) as a secondary metric, for comparability.
- Stage 3 candidate: an explicit-diversity condition (a BMD-style mutual-information bonus) to compare implicit and explicit preservation directly.
- Hypotheses and primary endpoints are unchanged; the pre-registration draft proceeds.

## Addendum, 27 October 2026: quality-diversity and repertoire-based recovery

**Why this was checked.** Whether to add a quality-diversity condition (MAP-Elites and related methods), and whether the project is positioned correctly against that literature. Searched on 27 October 2026, then audited by a separate agent that had not seen the reasoning; it read Cully et al. (2015), DDiffPG, SMERL, DOMiNO and Quality-Diversity Actor-Critic in full text. The DDiffPG result was re-checked independently.

**Correction to gap 2 above.** The first pass treated "repertoire → adaptation" as open apart from SMERL. That was too generous. A diverse repertoire is already known to aid recovery when the repertoire is explicit and can be selected from:

| Work | What it shows | Consequence |
| --- | --- | --- |
| [Cully, Clune, Tarapore & Mouret, 2015](https://arxiv.org/abs/1407.3501) (*Nature*, "Robots that can adapt like animals") | MAP-Elites builds a map of about 13,000 hexapod gaits in simulation before deployment. After damage, Bayesian optimisation searches the map with real trials and recovers in about a minute, with no diagnosis and no gradients. Searching inside the map beats searching raw parameters by a wide margin. | Landmark result that a repertoire buys recovery. It did not vary repertoire size or diversity as a controlled factor, and it recovers by search with a performance signal, not by relearning. |
| [Allard, Smith, Chatzilygeroudis & Cully, 2022](https://arxiv.org/abs/2204.05726) (GECCO, Hierarchical Quality-Diversity for Online Damage Recovery) | Extends repertoire-based recovery to a damaged hexapod navigating a maze; states that adaptability depends on the diversity of the repertoire. | Cite with Cully et al. |
| [Kumar et al., 2020](https://arxiv.org/abs/2010.14484) (SMERL, NeurIPS); [Zahavy et al., 2023](https://arxiv.org/abs/2205.13521) (DOMiNO, ICLR) | A set of diverse near-optimal policies, selected after a perturbation, beats a single policy (obstacles, forces, motor failures). | Already cited (SMERL); DOMiNO added. |
| [Grillotti, Faldor, González León & Cully, 2024](https://arxiv.org/abs/2403.09930) (Quality-Diversity Actor-Critic, ICML) | One skill-conditioned policy; few-shot skill selection under hurdles, motor failure, gravity and friction changes. | The single-policy form of quality-diversity. |
| [Li, Krohn, Chen, Ajay, Agrawal & Chalvatzaki, 2024](https://arxiv.org/abs/2406.00681) (DDiffPG, NeurIPS) | **Closest.** Random obstacles block paths in AntMaze. A planner switches the learned policy to another mode after 10 stalled steps; it reaches 1.0 success against 0.0–0.5 for single-mode baselines. Described by its authors as a proof of concept. | The blocked-route recovery test this project plans, done with explicit, selectable modes. |

**What stays open.**
1. Whether choosing which goals to practise preserves a repertoire *implicitly* (gap 1, unchanged). Nothing found does this.
2. A narrower recovery question: whether a repertoire that lives inside one goal-conditioned policy, with no latent and no mechanism for selecting a mode, speeds recovery under a common gradient-based adaptation procedure. The work above recovers by *selecting* among indexed modes.
3. Under sampled actions, recovery is expected to follow largely from preserved repertoire. What could make it diverge is the quality of the preserved minority route (how often and how quickly it reaches the goal), which this project should therefore measure.

**Quality-diversity as a condition: not in the first experiment.**
- A population of policies cannot be scored on the single-policy repertoire measure, run through the common adaptation procedure, or used as a base policy for `switch-or-recalibrate`.
- Frontier novelty is already the condition closest to quality-diversity: selecting the least-visited achieved goal is novelty search over end positions, the family that groups goal exploration processes with novelty search and quality-diversity ([Colas, Sigaud & Oudeyer, 2018](https://arxiv.org/abs/1802.05054); [Forestier et al., 2022](https://jmlr.org/papers/v23/21-0808.html)).
- With one goal per fixed-length episode, archive-based goal selection reduces to frontier novelty (if keyed by end position) or merely reweights which task goals are practised (if keyed by how they were reached). A distinct version needs two-goal episodes, a waypoint and then the task goal, as in Go-Explore ([Ecoffet et al., 2021](https://arxiv.org/abs/2004.12919)).
- At an equal environment-step budget, population-based quality-diversity is starved: MAP-Elites runs typically use far more evaluations than this project's budget.

**Candidates recorded for Stage 3**, chosen after the first experiment:
- Diversity in goal selection: two-goal archive-based selection (above). Its behaviour descriptor (a coarse via-cell) nearly coincides with route class in M1, so it would be a targeted upper reference, not evidence that quality-diversity preserves repertoire.
- Diversity in the objective: an adaptation of BMD. BMD adds a mutual-information bonus to a reward and fine-tunes with PPO; GCSL has no reward, so this needs a latent-conditioned learner.
- Quality-diversity as an experience source: a MAP-Elites collector gathers data and the unchanged learner trains on it, as with random actions. Exploratory only.

Single-policy quality-diversity is the relevant comparison for this project, not a population of elites: see [Faldor, Chalumeau, Flageat & Cully, 2023](https://arxiv.org/abs/2303.03832) (archive distillation into one descriptor-conditioned policy) and [Batra, Tjanaka, Nikolaidis & Sukhatme, 2024](https://arxiv.org/abs/2407.17515), who argue for one goal-conditioned policy rather than an archive in maze tasks. MAP-Elites itself is [Mouret & Clune, 2015](https://arxiv.org/abs/1504.04909).

**Measurement changes that follow** (in the specification): zero-shot recovery measured with sampled actions, since under greedy actions a policy that prefers the blocked route takes it every time whatever its repertoire; a "best of m sampled attempts" recovery measure, comparable with the selection-based work above; and per-route quality (success and time to goal for each route class).

**Venue note.** BMD (Longhini et al., arXiv 2605.11387) appears at ICML 2026 under the title "Unsupervised Mode Discovery for Fine-tuning Multimodal Generative Policies".

## How this was checked, and its limits

Searched on 22 October 2026 with web search, then audited by a separate agent that had not seen the search. The audit re-read every cited source, corrected four overstated claims, and ran its own search for a paper that would fail the gate; it found Polychromic Objectives (added above) and nothing that runs this comparison. Near misses checked and set aside: [MAGELLAN](https://arxiv.org/abs/2502.07709) (learning-progress goal selection for LLM agents; no demonstrations, no policy-repertoire or environment-change tests).

This is a first pass, not a systematic review. Several papers were read from abstracts and summaries rather than full text. Flowers/Inria and Pitis/Ba follow-ups and papers citing BMD had one pass each. BMD is recent and few papers cite it yet, so a repeat check against its citations is due before the Stage 2 pre-registration is frozen.
