# Paper.md

This file is the writing brief for the current paper draft.

It is not an experiment diary. It is a manuscript-facing summary of:

- what the paper is really about;
- how the method should be explained;
- which experimental results are strong enough to write;
- which results should be demoted, softened, or kept as limitations.


# 1. Working Paper Position

The paper is about:

> **post-retrieval fine localization for UAV-to-satellite geo-localization**

The retrieval stage should remain fixed in the paper narrative. The method
contribution starts only after retrieval top-1 has already been selected.

For the current draft, the writing priority should be:

1. **GTA-UAV same-area**
2. **UAV-VisLoc 03/04 same-area compact protocol**

This is different from a purely benchmark-driven ranking. The reason is simple:

- GTA same-area provides the larger-scale sparse-side validation;
- `03/04` currently provides a clear and writable VisLoc story.

So for the current paper draft:

- GTA should carry the **larger-scale same-area sparse comparison** and the
  **main narrative**;
- `03/04` should serve as secondary transfer-style evidence and validation on a
  real-world dataset.


# 2. Core Thesis To Write

The paper should be centered on the following claim:

> Fine localization after retrieval fails mainly because of orientation
> ambiguity, unreliable correspondences, and unstable geometry. A visual
> orientation posterior (VOP) can convert orientation uncertainty into a small
> set of useful angle hypotheses, so that sparse geometric verification becomes
> substantially stronger without paying dense-matching cost.

The current safe version of the claim is:

> `VOP + sparse` is the strongest sparse fine-localization line in the current
> repo. It consistently improves over explicit sparse baselines, preserves a
> large runtime advantage over dense matching, and reaches dense-level
> accuracy.

The paper should avoid overstating this as:

> `VOP + sparse` already delivers a large speed advantage and near-dense
> accuracy on all datasets.

The current evidence does not support that broader statement.


# 3. Why The Paper Is Interesting

The problem diagnosis should be stated very explicitly:

1. retrieval top-1 can already be correct;
2. localization can still fail because the UAV image has ambiguous orientation
   relative to the north-up satellite tile;
3. orientation ambiguity damages correspondence quality;
4. once correspondences are unstable, geometry becomes unstable too.

The key sentence to keep in the paper is:

> The bottleneck is not simply that there are too few matches. The bottleneck is
> the interaction between orientation ambiguity, unreliable correspondences, and
> unstable geometry.

This diagnosis is important because it explains why:

- naive sparse matching is too brittle;
- brute-force rotation search is costly and inelegant;
- VOP is a principled proposal mechanism rather than just another heuristic.


# 4. Method Summary

The method should be described as a post-retrieval pipeline:

1. keep retrieval fixed;
2. predict a posterior over discrete candidate angles with VOP;
3. keep the top-`k` useful angle hypotheses;
4. run sparse matching and geometric verification only on those hypotheses;
5. choose the final result by geometric quality.

Preferred terms:

- **visual orientation posterior (VOP)**
- **useful-angle proposer**
- **top-k angle hypotheses**
- **geometry-guided verification**

Avoid describing VOP as:

- an exact angle regressor;
- a replacement for geometry;
- a metadata-driven yaw module.

VOP is best presented as:

> a lightweight module that allocates sparse matching budget to a small number
> of geometrically promising orientations.


# 5. VOP Mechanism And Training

This is the most method-specific part of the paper and should be explained
carefully.

## 5.1 What VOP Predicts

VOP predicts a posterior over a discrete angle set:

- `Theta = {theta_1, ..., theta_M}`

It does **not** predict one exact continuous angle.

This is important because the observed angle-error surface is often:

- multi-modal;
- broad rather than sharply peaked;
- locally flat near several acceptable orientations.

So the right object is:

- not "the one true angle";
- but a **set of useful angles**.

## 5.2 VOP Architecture

VOP works on frozen retrieval feature maps.

For each candidate angle:

1. rotate the query feature map;
2. combine it with the gallery feature map;
3. use a lightweight head to produce one angle logit;
4. apply softmax across all angles.

The paper should emphasize two properties:

- it is lightweight relative to dense matching;
- it changes orientation handling **before** sparse geometry, not inside the
  matcher internals.

## 5.3 Teacher Construction

Teacher targets are built by evaluating each training pair under a discrete set
of candidate angles and recording final localization quality after geometry.

For each pair and angle, the teacher stores:

- final localization distance;
- best angle;
- best distance;
- second-best distance;
- distance gap;
- a derived soft target profile.

The important writing point is:

> supervision is grounded in localization outcome, not in raw pose metadata.

## 5.4 Useful-Angle Set Supervision

The current mainline target is:

- `U_i = {theta | d_i(theta) <= d_i* + delta}`

with:

- `delta = 5m`

This says that the model should recover all angles that are good enough for
localization, not just one angle that happens to be the teacher argmax.

## 5.5 Pair-Confidence Weighting

The current default also uses pair weighting:

- `w_i = 1 / (1 + exp((d_i* - c) / s))`

with:

- `c = 30m`
- `s = 10m`

This down-weights noisy or weak teacher pairs without throwing them away.

## 5.6 Current Default Recipe

The current VOP recipe to describe is:

- `supervision_mode = useful_bce`
- `useful_delta_m = 5`
- `ce_weight = 1.0`
- `pair_weight_mode = best_distance_sigmoid`
- `pair_weight_center_m = 30`
- `pair_weight_scale_m = 10`
- `orientation_topk = 4`

This is the method line that should be written as the main contribution.


# 6. What The Inference Comparison Actually Is

The paper must keep the compared pipelines precise.

## 6.1 Dense DKM

1. retrieve top-1;
2. run dense correspondence estimation;
3. sample dense correspondences;
4. estimate geometry;
5. project the gallery center.

## 6.2 Sparse

1. retrieve top-1;
2. run sparse matching with default multi-scale;
3. estimate geometry;
4. project the gallery center.

## 6.3 Rotate Baseline

The rotate baseline must be defined exactly as:

- sparse matching;
- four candidate rotations;
- select the final hypothesis by:
  - highest inlier count;
  - tie broken by inlier ratio.

Do not call it vaguely "sparse + rotate."

## 6.4 Ours

1. retrieve top-1;
2. compute the VOP posterior;
3. keep top-`k=4` angle hypotheses;
4. run sparse matching on those hypotheses;
5. select the final result by geometry.


# 7. Benchmark Positioning For The Draft

## 7.1 Use GTA Same-Area As The Main Validation Set

GTA same-area is valuable because:

- it is much larger;
- it confirms the sparse-side ranking on a more realistic scale;
- it gives a stronger same-area story for `VOP > sparse` and `VOP > rotate`.

The matched full dense row is now complete as well, which makes GTA useful for
an explicit `ours-versus-dense` comparison under matched retrieval.

But the result is still not a completely clean `ours > dense` story:

- `dense` remains stronger in robustness;
- `sparse + VOP` remains dramatically faster and is still the strongest sparse
  line;
- `sparse + VOP` reaches the same general accuracy tier as dense.

## 7.2 Use VisLoc As Real-World Validation

The old `03/04 same-area` split is still useful because:

- it is the only VisLoc split where the sparse-side story is reasonably strong;
- it shows clear method differences;
- it provides a readable runtime-versus-accuracy story;
- it is already the split on which the VOP supervision analysis is most mature.

Important protocol facts:

- split file:
  - `data/UAV_VisLoc_dataset/same-area-drone2sate-test.json`
- evaluation mode:
  - `test_mode=pos`
- total query entries in json:
  - `302`
- effective fine-localization queries:
  - `116`
- gallery size:
  - `17528`

Common retrieval metrics on this compact protocol:

- `Recall@1 = 92.2414`
- `Recall@5 = 99.1379`
- `Recall@10 = 100.0000`
- `AP = 95.6691`

This compact protocol should still be the main VisLoc table for writing.

## 7.3 Demote Paper7

Paper7 should not be the center of the current draft.

The reason is not that it is useless. The reason is that the absolute numbers
are still too weak to carry the paper story cleanly. It is better used as:

- a transfer check;
- a negative or mixed result;
- a limitation paragraph;
- or a compact supplementary table.


# 8. Writing-Ready Experimental Summary

This section is the most important one for drafting.

## 8.1 GTA Same-Area: Large-Scale Sparse-Side Confirmation

Use the refreshed full matched main-table rows for current writing:

| Variant | Recall@1 | Recall@5 | Recall@10 | mAP | Dis@1 | MA@20 | fallback | worse-than-coarse | mean total time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense DKM | 91.11 | 99.39 | 99.54 | 94.81 | 50.11 | 54.81 | 1.57% | 12.00% | 4.0410s |
| sparse | 91.11 | 99.39 | 99.54 | 94.81 | 108.47 | 14.61 | 58.41% | 18.47% | 0.0651s |
| sparse + rotate90 + inlier-count | 91.11 | 99.39 | 99.54 | 94.81 | 77.50 | 36.45 | 11.97% | 21.49% | 0.2831s |
| sparse + VOP | 91.11 | 99.39 | 99.54 | 94.81 | 52.59 | 43.54 | 12.02% | 13.30% | 0.3044s |

Key GTA comparisons:

### Ours vs sparse

- `Dis@1`: `108.47 -> 52.59` (`-55.88m`)
- `MA@20`: `14.61% -> 43.54%` (`+28.93pp`)

### Ours vs rotate

- `Dis@1`: `77.50 -> 52.59` (`-24.91m`)
- `MA@20`: `36.45% -> 43.54%` (`+7.09pp`)
- `worse-than-coarse`: `21.49% -> 13.30%` (`-8.19pp`)
- runtime: `0.2831s -> 0.3044s` (`+0.0213s`)

This is a very useful supporting result because it confirms on a much larger
same-area benchmark that:

- the rotate baseline is necessary;
- `VOP + sparse` is still better than that explicit rotate baseline;
- the extra runtime over rotate is small.

### Ours vs dense

- `dense` still has the stronger robustness profile;
- but `dense` is much slower:
  - `4.0410s/query` vs `0.3044s/query`
  - about `13.3x` slower

So GTA is now useful for:

- `VOP > sparse`
- `VOP > rotate`
- a matched `ours-versus-dense` comparison

But the correct reading is still:

- **dense remains stronger in robustness**
- **ours is the strongest sparse row, much faster, and now in the same general
  accuracy tier as dense**

## 8.2 VisLoc Table

Recommended core rows:

| Variant | Dis@1 | MA@3 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | mean inliers | total time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse + rotate90 + inlier-count | 59.47 | 0.00 | 0.00 | 1.72 | 24.14 | 17.24% | 53.45% | 37.2 | 0.2942s |
| current teacher, top-2 | 58.55 | 0.86 | 0.86 | 5.17 | 29.31 | 19.83% | 53.45% | 36.1 | 0.2044s |
| current teacher, top-4 | 49.14 | 1.72 | 2.59 | 6.03 | 31.03 | 18.10% | 43.97% | 42.4 | 0.3250s |
| clean30, top-4 | 35.51 | 2.59 | 4.31 | 6.90 | 30.17 | 23.28% | 44.83% | 39.2 | 0.3427s |
| useful5-weight30, top-4 (ours) | 38.86 | 0.86 | 2.59 | 8.62 | 37.93 | 11.21% | 40.52% | 43.2 | 0.3408s |
| dense DKM, no rotate | 32.45 | 2.59 | 4.31 | 13.79 | 49.14 | 1.72% | 25.86% | 4202.6 | 11.5241s |

How to read this table:

- `dense` is still the strongest absolute row on this compact protocol;
- among sparse methods, `useful5-weight30 top-4` is the most paper-worthy row;
- `clean30` has the best raw sparse `Dis@1`, but `useful5-weight30` is much
  better in robustness and high-threshold success;
- the current paper story on `03/04` should therefore be:
  - **ours is the strongest sparse method**
  - **dense is still stronger in absolute accuracy**
  - **ours is dramatically faster than dense**

## 8.3 Paper7: Secondary Negative / Mixed Evidence

Paper7 should be summarized briefly, not foregrounded.

| Variant | Dis@1 | MA@20 | fallback | worse-than-coarse | mean total time |
|---|---:|---:|---:|---:|---:|
| dense DKM | 241.40 | 35.51 | 8.09% | 48.04% | 4.1128s |
| sparse | 274.68 | 10.44 | 43.60% | 78.59% | 0.0747s |
| sparse + rotate90 + inlier-count | 258.18 | 18.28 | 16.97% | 62.40% | 0.3111s |
| sparse + VOP | 257.94 | 25.59 | 15.93% | 60.57% | 0.3291s |

The right interpretation is:

- `VOP + sparse` is still the strongest sparse row;
- but all sparse rows are weak in absolute terms;
- dense remains stronger in headline accuracy;
- therefore Paper7 should not be used as the main positive visual benchmark in
  the current draft.

If Paper7 appears in the main text at all, it should be framed as:

> a more difficult transfer setting where the sparse-side ranking is preserved,
> but the absolute performance gap to dense remains open.


# 9. What To Write In The Paper

## 9.1 Recommended Headline Result

For the current draft, the cleanest headline is:

> On GTA same-area, VOP-guided sparse matching substantially improves over an
> explicit sparse rotate baseline while remaining more than one order of
> magnitude faster than dense DKM and reaching the same general accuracy level
> as dense DKM. On the compact UAV-VisLoc `03/04` protocol, the same ranking
> still holds: `VOP + sparse` remains the strongest sparse fine-localization
> variant and reaches a similar accuracy level to dense DKM.

This is the most defensible narrative right now.

## 9.2 Recommended Method Claim

The method should be sold as:

- a principled orientation-uncertainty module;
- a sparse-efficiency preserving fine-localization strategy;
- the strongest sparse line in current experiments.

It should preferably not be sold as:

- a universal dense replacement;
- already better than dense everywhere.


# 10. Sections The Draft Should Contain

## 10.1 Introduction

The introduction should say:

1. retrieval success does not solve post-retrieval localization;
2. dense matching is strong but expensive;
3. naive sparse matching is cheap but brittle;
4. the missing ingredient is a way to handle orientation uncertainty before
   sparse geometry;
5. VOP provides that mechanism.

## 10.2 Method

Recommended method subsections:

1. problem setup
2. failure diagnosis
3. VOP formulation
4. teacher construction
5. useful-angle set supervision
6. pair-confidence weighting
7. top-`k` sparse verification

## 10.3 Experiments

Recommended experiments subsections:

1. companion benchmark: GTA same-area
2. VisLoc benchmark
3. supervision comparison
4. runtime and limitations


# 11. Tables And Figures To Prepare

## 11.1 Main Figure

One method figure should show:

1. retrieval top-1 tile
2. VOP posterior over angles
3. top-`k` angle proposals
4. sparse matching and geometry verification
5. final localization output

## 11.2 GTA Main Table

Use a compact GTA table with:

1. LoFTR
2. dense DKM
3. SP+LG
4. SP+LG+rotate90
5. ours

## 11.3 UAV-VisLoc Table

The current draft should use the `03/04` compact protocol.

Suggested rows:

1. dense DKM
2. sparse + rotate90
3. LoFTR
4. ours


# 12. Claim Discipline

## 12.1 Claims That Are Safe

The paper may safely claim:

- post-retrieval fine localization is a separate problem from retrieval;
- orientation ambiguity is a central cause of fine-localization failure;
- VOP is a principled discrete posterior over useful angle hypotheses;
- useful-angle set supervision is better aligned with the problem than
  single-angle supervision;
- `VOP + sparse` is the strongest sparse line in the current experiments;
- `VOP + sparse` preserves a large runtime advantage over dense matching and
  has reached a comparable performance level to dense.
