# Human route check — 28 October 2026

The author supplied all 60 labels in chat after viewing the blind sheet.
Before any classifier answers were displayed, the instructions were clarified:
crossing either direction counts, a touch and reversal does not, and the final
crossing chooses the arm. A dashed midpoint marker was added for visibility.
The 60 trajectory IDs and positions were not changed.

The supplied line is preserved verbatim in
`results/stage1/route_handcheck/human_comparison.json`.

- Certain labels: 56. Agreement: **56/56**.
- Certain north: 21; south: 24; neither: 11. No certain disagreements.
- Uncertain IDs: **13, 35, 48, 59**. All four match too; they remain excluded
  from the certain-agreement numerator and denominator.
- Comparison uses the exact rounded positions shown in the sheet. Fingerprints
  bind the evidence to those positions and the classifier/maze source files.
- This is a challenge set selected to include difficult cases; it is not a
  representative accuracy estimate on future learner rollouts.

The minimum of 50 certain manually checked trajectories is met. This report is
the labelling evidence, **not** the repository's separate independent code audit.
