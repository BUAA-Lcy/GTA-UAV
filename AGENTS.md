# AGENTS.md

This file is the long-term research and execution playbook for this repository.

Its purpose is not only to help with coding, but to guide the agent toward
**paper-worthy progress** on the **fine localization stage after retrieval**
for UAV-to-satellite / cross-view geo-localization.

The agent must prioritize:
1. research clarity,
2. methodological elegance,
3. reproducible experimentation,
4. honest interpretation,
over shallow metric chasing.


# 1. Project Scope

This repository contains a two-stage geo-localization pipeline:

1. **Retrieval stage**: retrieve top-K satellite candidates for each UAV query.
2. **Fine localization stage**: refine localization using image matching / geometric verification / orientation-aware refinement after retrieval.

The current research focus is:

> **Fine localization after retrieval**  
> not retrieval training, not backbone redesign, unless explicitly requested.

The working problem is:

> Post-retrieval fine localization is fragile because of
> orientation ambiguity, unreliable correspondences, and unstable geometry.

The current research target is to find a **cleaner, more academic solution**
than brute-force rotation enumeration plus heuristic post-selection.


# 2. Repository Facts That Must Be Respected

These are current repository facts and should not be silently violated.

- UAV-VisLoc preprocessing is expected to go through `scripts/prepare_dataset/visloc.py`.
- Raw pose fields (`Phi1`, `Phi2`, `Omega`, `Kappa`, etc.) should be preserved as metadata rather than silently redefined.
- In the current VisLoc evaluator, fine localization is applied only in `D2S` mode and only on the retrieved top-1 gallery image.
- In the current stable VisLoc convention, when `--use_yaw` is enabled, the query UAV image is rotated by `-Phi1` against a north-up satellite image.
- The current sparse fine-localization matcher already supports heuristic rotation search and an optional second phase.
- Retrieval quality and fine-localization quality are different:
  - retrieval metrics: `Recall@K`, `AP` / `mAP`
  - fine localization metrics: `Dis@K`, matched location quality, fallback rate, runtime
- The current VisLoc path is mainly controlled by:
  - `Game4Loc/eval_visloc.py`
  - `Game4Loc/game4loc/evaluate/visloc.py`
  - `Game4Loc/game4loc/dataset/visloc.py`
  - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
  - `scripts/prepare_dataset/visloc.py`

If any new method changes orientation handling, do it in the evaluator / matcher / model layer,
not by silently changing raw metadata semantics.


# 3. Execution Environment

- Use the `gtauav` Conda environment for experiments that are intended to be comparable.
- Keep commands reproducible.
- If environment-specific details are needed on the current workstation, document them clearly in scripts or local notes rather than hardcoding them as universal assumptions.
- Do not mix interpreters across experiments that will be compared.


# 4. Core Research Goal

The goal is **not** to become a larger collection of heuristics.

The goal is to produce a method that can be defended in a paper as a structured solution to a real failure mode.

The current working hypothesis is:

> Fine localization fails mainly not because there are too few matches,
> but because the system suffers from:
> - latent orientation ambiguity,
> - low-quality correspondences,
> - and unstable geometric estimation.

Therefore, the most promising direction is:

> model or score orientation explicitly,
> improve or condition correspondences,
> and estimate confidence for geometric acceptance,
> instead of depending on brute-force search plus ad hoc thresholds.


# 5. Research Priorities

When choosing what to do next, optimize in this order:

1. **Methodological elegance**
2. **Scientific clarity**
3. **Reliable fine-localization improvement**
4. **Reasonable runtime**
5. **Low implementation complexity**
6. **Minimal disruption to retrieval stage**

Do not optimize only for a slightly better metric if the method becomes a messy heuristic soup.


# 6. What the Agent Should Focus On

The agent should mainly help with **fine localization research** in four modes:

## A. Research design
- identify methodological weaknesses of the current fine localization pipeline;
- propose cleaner alternatives to brute-force multi-angle search;
- distinguish real method contributions from weak heuristic patches;
- convert empirical observations into explicit hypotheses.

## B. Implementation
- add or modify only the fine localization stage unless explicitly asked otherwise;
- keep changes modular, small, and reversible;
- expose important hyperparameters clearly;
- add logging for new orientation scores, confidence scores, or gating decisions.

## C. Experimentation and analysis
- run controlled ablations;
- compare under the same split, same checkpoint, same retrieval backbone;
- separate gains in retrieval from gains in fine localization;
- identify failure modes, not just success cases.

## D. Paper support
- help write contribution claims, ablation logic, failure analysis, method framing, and result summaries;
- never write the paper as “we tried many tricks and found one that works”;
- instead frame it as:
  - a failure diagnosis,
  - a method principle,
  - a structured solution,
  - and evidence.


# 7. What the Agent Must NOT Do

Do NOT:

- blindly add more brute-force orientation branches;
- keep stacking rescue stages without strong justification;
- treat RANSAC threshold tuning as the main research direction;
- silently modify retrieval stage when the task is fine localization only;
- claim “SOTA” if the evaluation protocol does not exactly match the reference;
- compare subset results against full-protocol baselines as if they were directly equivalent;
- present retrieval metric gains as fine-localization gains or vice versa;
- rewrite raw pose metadata semantics to make one experiment look better;
- keep adding one-off sweep flags and manual offsets as permanent defaults;
- overhype weak results.

If a proposal is only another heuristic patch, say so explicitly.


# 8. Preferred Research Directions

When proposing next-step ideas, prioritize these directions.

## Tier 1: Most promising and realistic
1. **Orientation scoring / prediction**
   - predict a small number of likely orientations;
   - reduce or replace exhaustive rotation search.

2. **Pose-conditioned matching**
   - use yaw / coarse orientation / candidate direction as a condition for correspondence generation, filtering, or reweighting.

3. **Learnable confidence / acceptance estimation**
   - predict whether the estimated geometry should be trusted;
   - replace hand-designed acceptance gates with a more principled model.

## Tier 2: Strong and still practical
4. **Feature reuse for multi-orientation scoring**
   - avoid repeated expensive matching for each tested angle;
   - extract once, score multiple hypotheses cheaply.

5. **Low-DoF geometric models before full homography**
   - compare similarity / affine-partial / affine against full homography;
   - determine whether lower-DoF geometry is more stable.

## Tier 3: Ambitious, use only if explicitly requested
6. **Geometry canonicalization / BEV-like normalization**
7. **Rotation-equivariant / steerable representations**
8. **Joint retrieval-localization-orientation modeling**
9. **Direct pose-estimation modules that replace part of the current retrieval-to-localization logic**

Do not jump to Tier 3 unless the user clearly chooses a larger redesign.


# 9. Standard Experimental Discipline

Every experiment must clearly record:

- exact code version or commit;
- dataset name and split;
- whether evaluation is full-set or subset;
- checkpoint path;
- query mode;
- retrieval backbone;
- matching mode (`sparse`, `dense`, etc.);
- orientation handling mode;
- geometric model;
- RANSAC / verification settings;
- acceptance criteria;
- runtime statistics.

Prefer one-factor-at-a-time ablations.

Do not silently change multiple things and then report the result as one idea.


# 10. Required Metrics for Fine Localization Experiments

At minimum, when matching is involved, report:

- `Recall@1`, `Recall@5`, `AP` / `mAP` if relevant;
- `Dis@1`, `Dis@3`, `Dis@5`;
- mean retained matches;
- mean inliers;
- mean inlier ratio;
- fallback rate;
- projection-out-of-bounds count;
- average matching time per query;
- count or proportion of cases where fine localization becomes worse than coarse retrieval.

If these are missing, the experiment is incomplete.


# 11. Required Baselines

For orientation-related fine localization experiments on the same checkpoint and split, include at least:

1. retrieval only;
2. fine localization without yaw prior;
3. current stable fine-localization baseline with `--with_match --sparse --use_yaw`;
4. the proposed method.

If the proposed method claims speed improvement, keep the retrieval backbone and evaluation protocol unchanged.


# 12. Standard Questions for Interpreting Results

For every new result, the agent must explicitly answer:

1. Does this truly improve fine localization, or just make some examples look better?
2. Does it hold on the full evaluation protocol, or only on a subset?
3. Does it improve `Dis@1` while damaging `Dis@3` / `Dis@5`?
4. Does it improve accuracy only by paying a large runtime cost?
5. Is the gain coming from better correspondences, or just looser acceptance?
6. Is this a cleaner method, or another heuristic stack?
7. Is the method promising as a paper contribution, or only as a baseline / ablation?

Do not leave these implicit.


# 13. Required Output Format for Experiment Summaries

Every completed experiment summary must use this structure:

## Experiment Name
- one-line purpose

## Change Compared to Baseline
- exact change only

## Quantitative Results
- main metrics
- runtime
- important counts/statistics

## Interpretation
- what likely improved
- what likely did not
- whether the result looks robust or fragile

## Decision
Choose exactly one:
- `KEEP`
- `REJECT`
- `NEEDS ONE FOLLOW-UP`
- `GOOD FOR PAPER ABLATION ONLY`

No vague endings.


# 14. Coding Rules

- Make minimal, localized changes.
- Do not rewrite the whole pipeline unless explicitly instructed.
- Keep the retrieval stage untouched unless the task explicitly includes retrieval.
- Preserve reproducibility.
- Avoid magic numbers; expose meaningful new thresholds or dimensions as named config arguments.
- Add logging for every newly introduced orientation score, confidence score, or gating signal.
- When implementing a new idea, also implement the simplest baseline needed to test whether it actually matters.

Modified Python files must at least pass syntax-level verification before handoff:
- `python -m py_compile ...`
- `python <entry_script> --help` for touched CLI entry points when relevant.


# 15. Paper-Writing Rules

When helping with the paper:

Do NOT frame the work as:
> “we tried many tricks and found one that works.”

Instead frame it as:

1. **Problem diagnosis**  
   Fine localization after retrieval is fragile because of orientation ambiguity and unreliable correspondences.

2. **Method principle**  
   Orientation should be modeled or scored, not only brute-force enumerated.

3. **Structured solution**  
   Use orientation-aware scoring / pose-conditioned matching / confidence-aware acceptance.

4. **Evidence**  
   Support claims with ablations, runtime-accuracy trade-offs, and failure analysis.

The agent must avoid exaggerated language such as:
- “novel”
- “state-of-the-art”
- “significant”
unless supported by explicit evidence under a matched protocol.


# 16. Interaction Style

The user does not want flattery.

Be direct, critical, and evidence-based.

- If an idea is weak, say it is weak.
- If a result is inconclusive, say it is inconclusive.
- If a direction is promising, explain exactly why.
- If a method is elegant but too risky for the current deadline, say so clearly.

Do not protect the user from negative conclusions.
Do not oversell weak evidence.


# 17. Default Task Pattern

When given a new task, proceed in this order:

1. restate the exact fine-localization question being addressed;
2. classify the task as one of:
   - baseline reproduction,
   - new method proposal,
   - ablation,
   - failure analysis,
   - paper-writing support;
3. propose the smallest rigorous next step;
4. implement or analyze it;
5. summarize whether it is worth keeping.

Prefer disciplined progress over chaotic exploration.


# 18. First-Class Candidate Ideas

If asked for the next elegant direction to try, start from these:

1. **Orientation scoring head**
   - predict a few likely directions;
   - match only those candidates.

2. **Pose-conditioned match filtering**
   - filter or reweight matches using orientation consistency.

3. **Learned localization confidence head**
   - predict whether the current geometric solution should be accepted.

4. **Feature reuse for multi-orientation evaluation**
   - avoid repeated full matching for every angle.

5. **Low-DoF geometry vs full homography**
   - test whether simpler geometry is more stable and cleaner.

Before inventing more brute-force variants, test these classes first.


# 19. Completion Criteria

A code change in this research thread is complete only if:

- it comes with at least one runnable command that reproduces the main result;
- the changed files pass basic syntax verification;
- the result clearly states whether the gain comes from retrieval, fine localization, or both;
- `Dis@1` and runtime are reported whenever matching is involved;
- the method is reproducible on the declared split;
- if it is proposed as a default fine-localization path, it must improve or preserve `Dis@1` without unjustified inference-time cost.

If a method only helps on a subset, say so clearly.


# 20. Final Principle

This project is not trying to become the best collection of angle sweeps and rescue heuristics.

It is trying to produce a **defensible, reproducible, and reasonably elegant**
method for post-retrieval fine localization.

Whenever there is a choice between:
- a slightly better but messier patch,
- and a cleaner method with a stronger research story,

prefer the cleaner method unless evidence strongly favors the patch.