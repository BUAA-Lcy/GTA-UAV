# Paper.md

This file is the paper-writing guide for this repository.

It is not an execution log. It is the place to keep the paper story clean,
stable, and aligned with the actual evidence in the repo.


# 1. Paper Goal

The paper is about:

> **post-retrieval fine localization for UAV-to-satellite geo-localization**

The paper is **not** about:

- improving retrieval;
- adding many independent tricks;
- brute-force orientation search as the main contribution;
- a new matcher backbone;
- using metadata yaw as the main solution.


# 2. Current One-Sentence Thesis

After retrieval, fine localization is fragile because orientation ambiguity,
unreliable correspondences, and unstable geometry interact with each other.
Instead of predicting one perfect orientation, it is better to propose a small
set of useful angle hypotheses and let geometry verification decide among them.


# 3. The Story The Paper Should Tell

## 3.1 Problem Diagnosis

The motivating diagnosis should be:

1. Retrieval may already place the correct satellite tile at top-1.
2. Fine localization can still fail because the UAV image orientation is
   ambiguous relative to the north-up satellite image.
3. This orientation ambiguity makes correspondences less reliable.
4. Once correspondences become unstable, geometric verification becomes fragile.

The core message is:

> the bottleneck is not simply "too few matches"

It is:

> orientation ambiguity interacting with unreliable correspondences and
> unstable geometry

## 3.2 Method Principle

The mainline method should be described as:

1. predict a small set of useful angle hypotheses;
2. run geometric verification under those candidate orientations;
3. let geometry select the final localization result.

Use the term:

- **useful-angle proposer**

More than:

- "orientation estimator"

because the current proposer should not be oversold as a perfectly calibrated
single-angle predictor.

## 3.3 Supervision Refinement

The paper-worthy supervision line is:

- **pair-confidence-weighted useful-angle set supervision**

The logic is:

1. teacher signals are noisy;
2. the angle-error surface is often not a single sharp mode;
3. supervision should therefore target a useful-angle set, not a single best
   angle only;
4. not all training pairs should contribute equally, so pair confidence should
   modulate the learning signal.


# 4. What To Emphasize And What To Avoid

## 4.1 Emphasize

- failure diagnosis before method design;
- top-k useful-angle proposal as the right structural choice;
- geometry verification as the final selector;
- noisy teacher supervision as an important practical issue;
- robustness metrics, not only raw `Dis@1`.

## 4.2 Avoid

- "we tried many heuristics and one worked";
- "the model predicts the exact orientation";
- "more matches automatically solve localization";
- "single-angle prediction is almost enough";
- any unmatched SOTA claim.


# 5. Evidence Hierarchy

Not all results in this repo have the same paper weight.

## 5.1 Highest-Value Evidence

- official evaluator results from `Game4Loc/eval_visloc.py`
- official evaluator results from `Game4Loc/eval_gta.py`
- robustness summaries:
  - `MA@3/5/10/20`
  - `fallback`
  - `worse-than-coarse`
  - `identity-H fallback`
  - `out-of-bounds`
  - `projection-invalid`
  - retained matches / inliers / inlier ratio
  - runtime

## 5.2 Mechanism Evidence

- cached top-k analysis
- useful-angle coverage
- oracle-best coverage
- supervision diagnosis runs

These are useful for explanation, but they should not become the headline
claim tables.

## 5.3 Development-Only Evidence

- UAV-VisLoc old `03/04` same-area split
- GTA same-area smoke runs

These are useful for diagnosis and iteration, but they must be labeled clearly.


# 6. Current Strongest Results Worth Citing

## 6.1 03/04 Dev Protocol: top-k beats single-angle

Official evaluator evidence already supports:

- `rotate=90`: `Dis@1 = 59.47m`, `MA@20 = 24.14%`
- `prior_topk=2`: `Dis@1 = 58.55m`, `MA@20 = 29.31%`
- `prior_topk=4`: `Dis@1 = 49.14m`, `MA@20 = 31.03%`

This supports the structural claim:

> top-k useful-angle hypotheses are better than forcing a single guess

But this is still dev-only evidence.

## 6.2 GTA Same-Area Supervision Comparison (`teacher_query_limit=2000`)

Run summary:

- `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/summary.md`

Official evaluator summary:

| Variant | Dis@1 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse baseline | 77.16 | 6.42 | 16.58 | 33.78 | 28.35% | 14.20% | 0.29% |
| Exp A current teacher | 70.27 | 8.31 | 19.20 | 39.44 | 19.40% | 14.00% | 0.32% |
| Exp B clean30 | 69.42 | 7.73 | 19.14 | 38.63 | 22.07% | 13.42% | 0.41% |
| Exp C weighted useful-angle | 63.03 | 8.74 | 21.64 | 42.17 | 17.51% | 11.56% | 0.15% |

This is currently the clearest GTA evidence that:

1. teacher noise matters;
2. hard filtering helps but is only a diagnostic baseline;
3. weighted useful-angle supervision is the best current GTA line.

## 6.3 GTA Same-Area Exp C Follow-Up

Run summary:

- `Game4Loc/work_dir/gta_exp_c_followup_runs/exp_c_followup_samearea_q2000_20260410/summary.md`

Official evaluator summary:

| Variant | Dis@1 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp C baseline (`delta=5`, `center=30`) | 63.03 | 8.74 | 21.64 | 42.17 | 17.51% | 11.56% | 0.15% |
| Exp C1 (`useful_delta_m=3`) | 63.82 | 8.74 | 20.94 | 42.23 | 16.67% | 12.69% | 0.17% |
| Exp C2 (`pair_weight_center_m=20`) | 62.72 | 8.31 | 21.49 | 41.94 | 19.05% | 10.75% | 0.23% |

Interpretation:

- the current Exp C default should stay:
  - `useful_delta_m = 5`
  - `pair_weight_center_m = 30`
- tightening the useful-angle set hurts more clearly;
- stricter pair weighting gives only a tiny raw `Dis@1` gain and is not more
  robust overall.


# 7. Current Default Configuration To Describe In Writing

When writing the current GTA same-area method, describe the default as:

- dataset:
  - GTA-UAV same-area
- evaluator:
  - `eval_gta.py`
- fine localization mode:
  - `--with_match --sparse`
- proposer usage:
  - `orientation_mode=prior_topk`
  - `orientation_topk=4`
- retrieval backbone checkpoint:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`
- current best GTA same-area VOP checkpoint:
  - `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/artifacts/exp_c_useful5_weight30_e6.pth`
- VOP supervision:
  - `useful_bce`
  - `useful_delta_m = 5`
  - `ce_weight = 1.0`
  - `pair_weight_mode = best_distance_sigmoid`
  - `pair_weight_center_m = 30`
  - `pair_weight_scale_m = 10`


# 8. Suggested Paper Outline

## 8.1 Abstract

The abstract should cover four moves:

1. retrieval is not the whole story;
2. post-retrieval fine localization fails under orientation ambiguity;
3. propose top-k useful-angle hypotheses plus geometry verification;
4. useful-angle set supervision with pair confidence improves robustness.

## 8.2 Introduction

Recommended arc:

1. UAV-to-satellite localization often studies retrieval, but accurate position
   refinement after retrieval remains fragile.
2. The difficulty is not merely correspondence quantity; it is ambiguity plus
   geometry instability.
3. Existing single-angle thinking is structurally mismatched.
4. Our method predicts a small useful-angle set and verifies geometrically.
5. We further refine supervision to account for noisy teacher signals.

## 8.3 Related Work

Organize around:

- UAV-to-satellite geo-localization;
- image matching and geometric verification;
- orientation estimation / pose priors;
- post-retrieval localization refinement.

## 8.4 Method

Suggested subsections:

1. problem setup;
2. failure diagnosis;
3. useful-angle proposal;
4. top-k geometric verification;
5. pair-confidence-weighted useful-angle set supervision.

## 8.5 Experiments

Separate the roles clearly:

1. official evaluator headline results;
2. supervision diagnosis;
3. mechanism analysis;
4. ablations that clarify the method but do not become the main claim.

## 8.6 Limitations

Be explicit about:

- current evidence being stronger on development and same-area settings than on
  a final large strict benchmark;
- proposer posteriors still being somewhat flat;
- current GTA VOP path being sparse-only.


# 9. Tables And Figures To Prepare

## 9.1 Must-Have Tables

1. Baseline vs `prior_topk` official evaluator table on the main benchmark.
2. Supervision comparison table:
   - sparse baseline
   - Exp A
   - Exp B
   - Exp C
3. Robustness table containing:
   - `MA@3/5/10/20`
   - `fallback`
   - `worse-than-coarse`
   - `out-of-bounds`
   - runtime

## 9.2 Must-Have Figures

1. Pipeline figure:
   - retrieval top-1
   - useful-angle proposal
   - geometry verification
   - final localization
2. Angle-error surface examples showing:
   - broad useful regions
   - multiple acceptable modes
3. Qualitative failure cases:
   - baseline fails
   - top-k useful-angle succeeds

## 9.3 Helpful Appendix Material

- teacher-noise diagnosis;
- clean-pair baseline as a diagnostic only;
- Exp C follow-up showing why the current default stays at
  `delta=5, center=30`.


# 10. Claim Discipline

The paper may reasonably claim:

- post-retrieval fine localization has a distinct failure mode from retrieval;
- top-k useful-angle proposal is better aligned with the problem than a single
  angle guess;
- pair-confidence-weighted useful-angle supervision is a better training
  direction than simple teacher distillation.

The paper should **not** currently claim:

- SOTA under a matched external benchmark;
- definitive superiority on the full final protocol unless that protocol is
  actually run;
- that the proposer itself recovers precise orientation faithfully.


# 11. Recommended Writing Language

Prefer wording like:

- "useful angle hypotheses"
- "post-retrieval fine localization"
- "geometry verification selects among candidate orientations"
- "teacher noise"
- "robustness metrics"

Avoid wording like:

- "we search many angles until one works"
- "the network predicts the exact orientation"
- "more correspondences alone solve localization"
- "our gains come from a stronger matcher"


# 12. Current Writing To-Do

If someone starts drafting now, the best order is:

1. write the Introduction and Method around the useful-angle story;
2. prepare the supervision comparison table from GTA same-area;
3. insert the Exp C follow-up as a short ablation confirming the chosen default;
4. clearly label 03/04 VisLoc as development-only evidence;
5. keep the final claim conservative until a larger strict protocol is run.


# 13. Bottom Line

The cleanest current paper line is:

> fine localization after retrieval is fragile because of orientation ambiguity
> and geometric instability; proposing a small set of useful angles and letting
> geometry verification choose among them is better than forcing a single
> orientation, and pair-confidence-weighted useful-angle supervision is the most
> defensible current training recipe.
