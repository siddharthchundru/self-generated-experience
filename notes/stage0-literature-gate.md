# Stage 0 literature gate — Project A

*Siddharth Chundru · 22 October 2026*

**Question for the gate.** Has anyone already run this comparison: one fixed goal-conditioned learner, starting from narrow demonstrations, with conditions that differ only in how goals are chosen during practice (task goals vs. frontier novelty vs. random), measuring competence, policy-level repertoire, and recovery after a route is blocked, plus a re-learning control that separates the data from the active loop?

**Verdict: pass, with a reframing.** I found no paper that runs this comparison. But one half of the central hypothesis — that task-directed improvement collapses behavioural modes — is now well documented in neighbouring settings, so it should be cited as motivation, not claimed as a finding. The contribution moves to the other half: whether *intrinsic goal selection*, with no explicit diversity objective, preserves repertoire, and whether preserved repertoire actually buys adaptation.

## What exists

| Work | What it shows | Overlap | Consequence |
| --- | --- | --- | --- |
| [Longhini et al., 2026](https://arxiv.org/abs/2605.11387) (BMD, ICML 2026) | RL fine-tuning of pre-trained generative policies (e.g. diffusion) collapses demonstrated modes into one reward-maximising mode. An intrinsic mutual-information reward over a discrete latent preserves them. Tested on a 2D four-mode navigation task, ManiSkill, ANYmal and Franka Kitchen; reports mode-weighted success, mode coverage and mode entropy. Perturbation tests (one locked joint, observation noise) on ANYmal only. | **Highest.** Mode collapse under task-directed improvement, measured with mode metrics, in a 2D navigation task. | Cite as the closest work. "Self-practice narrows repertoire" is no longer a headline. Adopt their mode-weighted success alongside route repertoire for comparability. |
| [Hamid et al., 2025](https://arxiv.org/abs/2509.25424) (Polychromic Objectives) | RL fine-tuning of pretrained policies collapses them onto a few easily exploited outputs; an objective that explicitly rewards diverse generations prevents this and generalises better under large perturbations. BabyAI, MiniGrid, Algorithmic Creativity. | **Medium–high.** Collapse, diversity and perturbation tests in grid mazes. | Diversity comes from an explicit objective with fixed goals, not from goal selection; no re-learning control. Cite with BMD. |
| [Du et al., 2024](https://arxiv.org/abs/2406.15917) ("To Err is Robotic") | At deployment, a demonstration-trained diffusion policy detects stalled progress and resamples among its own candidate actions, away from recorded failures, with no gradient updates. | **Medium–high, as motivation.** | Deployment-time recovery of this kind assumes the base policy still has alternative modes. Project A asks which ways of practising keep them. |
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

## Decision and changes

**Keep the project.** Reframe the lead claim:

- *Before:* "Self-practice narrows the repertoire, so it adapts worse than intrinsic exploration."
- *After:* "Task-directed improvement is known to collapse modes. Does intrinsically chosen practice preserve them without an explicit diversity objective — and does the preserved repertoire buy adaptation?"

Concrete changes:

- README: cite BMD and Polychromic Objectives as the closest work; restate the contribution as above.
- Metrics: add mode-weighted success rate (BMD's SRM) as a secondary metric, for comparability.
- Stage 3 candidate: an explicit-diversity condition (a BMD-style mutual-information bonus) to compare implicit and explicit preservation directly.
- Hypotheses and primary endpoints are unchanged; the pre-registration draft proceeds.

## How this was checked, and its limits

Searched on 22 October 2026 with web search, then audited by a separate agent that had not seen the search. The audit re-read every cited source, corrected four overstated claims, and ran its own search for a paper that would fail the gate; it found Polychromic Objectives (added above) and nothing that runs this comparison. Near misses checked and set aside: [MAGELLAN](https://arxiv.org/abs/2502.07709) (learning-progress goal selection for LLM agents; no demonstrations, no policy-repertoire or environment-change tests).

This is a first pass, not a systematic review. Several papers were read from abstracts and summaries rather than full text. Flowers/Inria and Pitis/Ba follow-ups and papers citing BMD had one pass each. BMD is recent and few papers cite it yet, so a repeat check against its citations is due before the Stage 2 pre-registration is frozen.
