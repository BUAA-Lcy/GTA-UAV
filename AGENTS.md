# AGENTS.md

This file is the research handoff and execution playbook for this repository.

It is written for agents who will continue the paper effort after the current
session. Read this file before running experiments or changing code.

The project goal is not "more tricks." The project goal is a defensible paper
about **post-retrieval fine localization** for UAV-to-satellite geo-localization.


# 1. Scope

This repository has two stages:

1. **Retrieval**
2. **Fine localization after retrieval**

The current paper focus is strictly:

> **fine localization after retrieval**

Do not change retrieval unless the user explicitly asks for retrieval work.

The current working diagnosis is:

> Fine localization fails mainly because of orientation ambiguity, unreliable
> correspondences, and unstable geometry, not simply because there are "too few
> matches."


# 2. Current Handoff Status

As of this handoff:

- Active branch: `codex/vop-experiment`
- Current uncommitted work exists in:
  - `AGENTS.md`
  - `Paper.md`
  - `Game4Loc/scripts/run_gta_same_area_exp_c_followup.sh`
- Those uncommitted additions now include:
  - GTA same-area Exp C minimal follow-up automation
  - updated research handoff notes
  - a paper-writing guide focused on the current method line
- Recent project capabilities already include:
  - supervision-diagnosis support in `train_vop.py`
  - GTA-UAV teacher-cache support in `build_vop_teacher.py`
  - GTA-UAV sparse VOP evaluation support in `eval_gta.py`
  - GTA-UAV sparse VOP inference integration in `game4loc/evaluate/gta.py`
- Do not revert user or prior-agent work unless explicitly requested.

Current research state:

1. **Single-angle VOP is not the right mainline.**
   - It was unstable.
   - It did not give a convincing paper story.

2. **The current mainline is:**
   - **top-k useful angle hypotheses + geometry verification**

3. **Current default proposer configuration:**
   - `prior_topk=4`

4. **But important warning:**
   - The old `03/04` same-area small protocol is now **development-only**.
   - Do **not** use it as the main benchmark for formal paper claims.

5. **Current supervision diagnosis status on the 03/04 dev protocol:**
   - `current teacher baseline`
   - `hard clean-pair filter baseline`
   - `pair-confidence-weighted useful-angle set supervision`
   have all been compared.

6. **Current best formal next step:**
   - Move the **useful-angle set supervision** idea to a larger / stricter
     protocol for formal validation.
   - Keep `hard clean-pair filter` only as a diagnostic baseline.

Current GTA-UAV migration status:

1. **VOP teacher building now supports GTA-UAV.**
   - `Game4Loc/build_vop_teacher.py` now accepts:
     - `--dataset gta`
   - GTA teacher distances are built in the dataset's planar `x/y` space using:
     - query `drone_loc_x_y`
     - gallery tile geometry from `game4loc.dataset.gta.sate2loc`

2. **GTA-UAV sparse official evaluation now supports VOP priors.**
   - `Game4Loc/eval_gta.py` now accepts:
     - `--orientation_checkpoint`
     - `--orientation_mode {off, prior_single, prior_topk}`
     - `--orientation_topk`
     - `--num_workers`
   - `Game4Loc/game4loc/evaluate/gta.py` now supports:
     - sparse `prior_single`
     - sparse `prior_topk`

3. **Current GTA-UAV priority path is sparse mode.**
   - Prefer:
     - `--with_match --sparse`
   - The GTA VOP integration is currently wired only for:
     - sparse fine localization
   - Do **not** treat dense-mode GTA evaluation as the current mainline for VOP.

4. **A same-area GTA smoke test already passed end-to-end.**
   - It covered:
     - GTA teacher cache build
     - GTA VOP training
     - GTA sparse `prior_topk=4` evaluation
   - Smoke artifacts:
     - `Game4Loc/work_dir/vop/gta_samearea_smoke_teacher.pt`
     - `Game4Loc/work_dir/vop/gta_samearea_smoke_vop.pth`
   - Important warning:
     - these smoke artifacts are **pipeline-validation only**
     - do **not** use their metrics as paper evidence

5. **Current environment note for GTA evaluation.**
   - On this machine, prefer:
     - `--num_workers 0`
   - Reason:
     - GTA eval with multi-process dataloading may raise:
       - `OSError: [Errno 95] Operation not supported`

6. **GTA-UAV evaluator now reports full robustness summaries.**
   - Current GTA official logs now include:
     - `MA@3m / MA@5m / MA@10m / MA@20m`
     - `fallback`
     - `worse-than-coarse`
     - `identity-H fallback`
     - `out-of-bounds`
     - `projection-invalid`
     - `mean_retained_matches`
     - `mean_inliers`
     - `mean_inlier_ratio`
     - `mean_vop_forward_time / mean_matcher_time / mean_total_time`
   - This was added as logging / summary only.
   - It does **not** change GTA evaluation semantics.

7. **A fixed-subset GTA same-area supervision comparison has completed.**
   - Protocol:
     - same-area
     - sparse
     - full same-area test
     - retrieval checkpoint fixed to GTA same-area official weight
     - teacher subset fixed to `2000` effective queries
     - `prior_topk=4`
   - Run summary:
     - `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/summary.md`
   - Main result:
     - baseline: `Dis@1 = 77.16m`
     - Exp A current teacher: `70.27m`
     - Exp B clean30: `69.42m`
     - Exp C weighted useful-angle: `63.03m`

8. **Current GTA default VOP training recipe is Exp C.**
   - Use:
     - `useful_bce`
     - `useful_delta_m = 5`
     - `ce_weight = 1.0`
     - `pair_weight_mode = best_distance_sigmoid`
     - `pair_weight_center_m = 30`
     - `pair_weight_scale_m = 10`
   - Keep Exp A / Exp B only as diagnostic baselines.

9. **A minimal GTA same-area Exp C follow-up has completed.**
   - Protocol fixed:
     - same-area
     - sparse
     - full same-area test
     - teacher subset fixed to `2000`
     - `prior_topk=4`
   - Run summary:
     - `Game4Loc/work_dir/gta_exp_c_followup_runs/exp_c_followup_samearea_q2000_20260410/summary.md`
   - Compared:
     - baseline Exp C:
       - `useful_delta_m = 5`
       - `pair_weight_center_m = 30`
     - Exp C1:
       - `useful_delta_m = 3`
     - Exp C2:
       - `pair_weight_center_m = 20`
   - Main result:
     - baseline Exp C remains the current default to keep
     - Exp C1 regresses more clearly
     - Exp C2 slightly improves raw `Dis@1` but weakens robustness
   - Practical conclusion:
     - the current GTA same-area line should keep:
       - `useful_delta_m = 5`
       - `pair_weight_center_m = 30`
     - if more GTA work is requested later, expand teacher subset before more
       micro-sweeps on these two knobs

10. **Current GTA `with_match` output behavior is log-only by default.**
    - `eval_gta.py --with_match --sparse` writes the standard app log under:
      - `Game4Loc/Log/...`
    - It does **not** dump per-query match visualization files by default.
    - Reason:
      - `GimDKM(..., sparse_save_final_vis=False)`
      - `SparseSpLgMatcher(..., save_final_matches=False)`
    - There is currently no CLI flag in `eval_gta.py` to expose this behavior.


# 3. Hard Constraints

Unless the user explicitly changes the scope, do **not**:

- modify retrieval training;
- redesign the retrieval backbone;
- silently reinterpret raw pose metadata (`Phi1`, `Phi2`, `Omega`, `Kappa`, etc.);
- claim "SOTA" under an unmatched protocol;
- compare subset numbers against full-protocol paper numbers as if directly comparable;
- add endless rescue heuristics;
- make brute-force angle sweeps the paper contribution;
- use yaw metadata / yaw parameters for the current visual-orientation line;
- rewrite matcher internals unless explicitly requested.


# 4. Repository Facts That Must Be Respected

- UAV-VisLoc preprocessing goes through:
  - `scripts/prepare_dataset/visloc.py`
- Current VisLoc evaluator applies fine localization only in:
  - `query_mode="D2S"`
  - retrieved top-1 gallery only
- Current stable yaw convention:
  - when `--use_yaw` is enabled, the query UAV image is rotated by `-Phi1`
    against a north-up satellite image
- GTA-UAV uses planar localization coordinates:
  - query location:
    - `drone_loc_x_y`
  - gallery tile geometry:
    - `game4loc.dataset.gta.sate2loc`
- Current main VisLoc path:
  - `Game4Loc/eval_visloc.py`
  - `Game4Loc/game4loc/evaluate/visloc.py`
  - `Game4Loc/game4loc/dataset/visloc.py`
  - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
  - `Game4Loc/game4loc/matcher/gim_dkm.py`
  - `scripts/prepare_dataset/visloc.py`
- Current main GTA-UAV fine-localization path:
  - `Game4Loc/eval_gta.py`
  - `Game4Loc/game4loc/evaluate/gta.py`
  - `Game4Loc/game4loc/dataset/gta.py`
  - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
  - `Game4Loc/game4loc/matcher/gim_dkm.py`
  - `Game4Loc/build_vop_teacher.py`
  - `Game4Loc/train_vop.py`

Current GTA-UAV evaluator facts:

- fine localization is currently applied on:
  - `query_mode="D2S"`
  - retrieved top-1 gallery only
- current VOP integration in GTA-UAV is currently:
  - sparse-only
  - retrieval-top1-only

If orientation handling changes, change it in evaluator / model / matcher layers,
not by changing metadata semantics.


# 5. Environment And Reproducibility

Use the same environment for comparable experiments:

- Conda env: `gtauav`
- Typical interpreter:
  - `/home/lcy/miniconda3/envs/gtauav/bin/python`

Execution rules:

- Run comparable experiments from:
  - `/home/lcy/Workplace/GTA-UAV/Game4Loc`
- Prefer:
  - `WANDB_MODE=disabled`
- Keep commands copy-paste reproducible.
- Do not mix interpreters across compared runs.
- For GTA-UAV evaluation on the current machine, prefer:
  - `--num_workers 0`


# 6. Evaluation Taxonomy

This project now has **three different evaluation roles**. Do not mix them.

## A. Official evaluator

Use:

- `Game4Loc/eval_visloc.py`
- `Game4Loc/eval_gta.py`

Purpose:

- headline results
- final `Dis@K`
- threshold success rates
- fallback / worse-than-coarse / runtime

This is the only source that should drive formal headline tables.

Dataset mapping:

- UAV-VisLoc:
  - `Game4Loc/eval_visloc.py`
- GTA-UAV:
  - `Game4Loc/eval_gta.py`

## B. Cached evaluator

Use:

- `Game4Loc/build_topk_cache.py`
- `Game4Loc/eval_topk_cached.py`

Purpose:

- mechanism analysis only
- oracle-best coverage
- useful-angle coverage
- covered-vs-missed error analysis

Do **not** use cached numbers as headline paper results.

Current limitation:

- cached top-k analysis is currently wired for:
  - UAV-VisLoc
- it is **not yet** ported to:
  - GTA-UAV

## C. Training supervision diagnosis

Use:

- `Game4Loc/train_vop.py`
- `Game4Loc/build_vop_teacher.py`

Purpose:

- understand whether supervision is noisy
- compare teacher variants
- compare useful-angle supervision schemes

This is development analysis, not final benchmark reporting.

Current support status:

- `build_vop_teacher.py` now supports:
  - UAV-VisLoc
  - GTA-UAV
- `train_vop.py` remains teacher-cache driven and is shared across both datasets


# 7. Dataset Protocols And What They Mean

## 7.1 03/04 same-area small protocol

This is the old protocol built from UAV-VisLoc areas `03` and `04`.

Use it only as:

- a **development protocol**
- a **supervision diagnosis protocol**

Do **not** use it as the main benchmark in the paper.

Important facts:

- Split generation is tied to `scripts/prepare_dataset/visloc.py`
- Same-area train/test are made only from areas `03` and `04`
- Local files currently used:
  - `data/UAV_VisLoc_dataset/same-area-drone2sate-train.json`
  - `data/UAV_VisLoc_dataset/same-area-drone2sate-test.json`
- Current official dev evaluation usually uses:
  - `--test_mode pos`

Under `--test_mode pos`:

- JSON total query entries: `302`
- Effective fine-localization evaluation queries: `116`
- Gallery size in current evaluator: `17528`

Why only `116` queries?

- The evaluator only keeps queries with non-empty `pair_pos_sate_img_list`

## 7.2 Expanded `pos_semipos`

This was tested once as a stress test.

Important conclusion:

- It greatly increases query count
- But it also changes the positive-label semantics
- Therefore it is **not** a clean drop-in replacement for `strict-pos`

Concrete result from that stress test:

- protocol: same-area, `test_mode=pos_semipos`
- query count increased from `116` to `302`
- but the metric semantics changed
- under that looser protocol:
  - `rotate=90`: about `90.68m`
  - `prior_topk=4`: about `101.17m`

Interpretation:

- this is not evidence against the top-k line itself
- it is evidence that `pos_semipos` should not replace strict-pos as the main
  benchmark without explicit discussion

Do not use `pos_semipos` as the formal benchmark unless the user explicitly
wants a looser protocol.

## 7.3 Formal benchmark target

The next agent should treat the real paper-facing benchmark as:

- a **larger / stricter UAV-VisLoc protocol**
- ideally expanded strict-pos
- not the old 03/04 development split

If a larger protocol is built, make sure:

- positive label semantics stay strict
- evaluation remains directly comparable
- 03/04 results are labeled clearly as development-only

## 7.4 GTA-UAV protocol status

Current local GTA-UAV protocol files:

- `data/GTA-UAV-data/same-area-drone2sate-train.json`
- `data/GTA-UAV-data/same-area-drone2sate-test.json`
- `data/GTA-UAV-data/cross-area-drone2sate-train.json`
- `data/GTA-UAV-data/cross-area-drone2sate-test.json`

Matched retrieval checkpoints:

- same-area:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`
- cross-area:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_cross_area.pth`

Current migration guidance:

- use same-area first for:
  - pipeline shakeout
  - supervision debugging
  - command validation
- then compare matched GTA settings separately:
  - same-area with same-area checkpoint
  - cross-area with cross-area checkpoint

Do **not**:

- mix same-area JSON with cross-area checkpoint
- mix cross-area JSON with same-area checkpoint
- quote smoke-test GTA metrics as evidence

Current smoke status:

- same-area smoke with `query_limit=2` has already run end-to-end
- this proves the GTA VOP path is wired
- it does **not** prove method quality


# 8. Current Method Line

## 8.1 Rejected / demoted lines

### Single-angle VOP

Status:

- **Rejected as main paper line**

Reason:

- unstable
- too close to "guess one angle"
- not aligned with the observed angle-error surface

### Confidence-aware verification

Status:

- **ablation only for now**

Reason:

- a lightweight verifier was tested
- it gave slight average improvement in one run
- but worsened `worse-than-coarse` / fallback behavior
- not clean enough to become the default second module yet

## 8.2 Current mainline

Current method principle:

> predict a small set of useful angle hypotheses, then let geometry verification
> decide among them

Current code-facing default:

- `orientation_mode=prior_topk`
- `orientation_topk=4`

Current proposer checkpoint:

- `Game4Loc/work_dir/vop/vop_0407_full_rankce_e6.pth`

Important interpretation:

- the proposer is not currently strong enough to be sold as a perfect
  orientation estimator
- it is better viewed as a **useful-angle proposer**


# 9. Current Experimental Conclusions

These are the most important conclusions that the next agent should not
re-derive from scratch unless there is a reason.

## 9.1 On the 03/04 dev protocol, top-k is better than single-angle

Official evaluator results already showed:

- `rotate=90`: `Dis@1 = 59.47m`, `MA@20 = 24.14%`
- `prior_topk=2` with current frozen VOP: `Dis@1 = 58.55m`, `MA@20 = 29.31%`
- `prior_topk=4` with current frozen VOP: `Dis@1 = 49.14m`, `MA@20 = 31.03%`

So:

- top-k useful-angle proposals are better than the simple four-angle baseline
- but this is still **development protocol evidence**

## 9.2 Angle-error surfaces are often not single sharp modes

Cached analysis showed many samples have:

- useful-angle intervals
- multiple useful modes
- more than one acceptable angle

Therefore:

- forcing single best-angle prediction is structurally mismatched

## 9.3 Current supervision diagnosis on 03/04 dev protocol

Three supervision variants were compared:

### A. Current teacher baseline

- checkpoint: `Game4Loc/work_dir/vop/vop_0407_full_rankce_e6.pth`

### B. Hard clean-pair filter baseline

- checkpoint: `Game4Loc/work_dir/vop/vop_0409_clean30_rankce_e6.pth`
- filter: keep only training pairs with `best localization error < 30m`

Observed interpretation:

- improves dev-protocol performance
- proves supervision noise matters
- but is still a crude threshold-based baseline
- likely biased toward easy pairs

### C. Pair-confidence-weighted useful-angle set supervision

- checkpoint: `Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- useful set:
  - `{theta | error(theta) <= best + 5m}`
- pair weight:
  - sigmoid based on `best_distance`

Observed interpretation:

- more aligned with the problem structure
- but the current minimal implementation learned an overly flat posterior
- still promising as the more paper-worthy direction

### Practical conclusion from the supervision diagnosis

If only one supervision idea should move to the larger strict-pos protocol:

- **take Exp C**:
  - pair-confidence-weighted useful-angle set supervision

Keep Exp B only as:

- diagnostic baseline
- evidence that teacher noise matters

### Most useful dev-only result table

Official evaluator:

| Variant | top-k | Dis@1 | MA@5 | MA@10 | MA@20 | worse-than-coarse | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| current teacher | 2 | 58.55 | 0.86 | 5.17 | 29.31 | 53.45% | 19.83% |
| current teacher | 4 | 49.14 | 2.59 | 6.03 | 31.03 | 43.97% | 18.10% |
| hard clean-pair `<30m` | 2 | 48.64 | 1.72 | 5.17 | 29.31 | 52.59% | 32.76% |
| hard clean-pair `<30m` | 4 | 45.51 | 4.31 | 6.90 | 30.17 | 44.83% | 23.28% |
| weighted useful-angle | 2 | 60.48 | 3.45 | 9.48 | 27.59 | 56.03% | 25.00% |
| weighted useful-angle | 4 | 48.86 | 2.59 | 8.62 | 37.93 | 40.52% | 11.21% |

Cached mechanism view:

| Variant | top-k | Oracle-best cov | Useful@+5m cov | Useful@+5m recall |
|---|---:|---:|---:|---:|
| current teacher | 2 | 0.1379 | 0.3103 | 0.1024 |
| current teacher | 4 | 0.1897 | 0.4397 | 0.2112 |
| hard clean-pair `<30m` | 2 | 0.1121 | 0.3103 | 0.1226 |
| hard clean-pair `<30m` | 4 | 0.2328 | 0.4569 | 0.2532 |
| weighted useful-angle | 2 | 0.0862 | 0.2845 | 0.0874 |
| weighted useful-angle | 4 | 0.1638 | 0.4138 | 0.1745 |

Interpretation of that table:

- Exp B currently gives the best raw dev-protocol result, but remains a crude
  hard-filter baseline
- Exp C is more structurally aligned with the problem, but the current minimal
  implementation is not yet stronger than Exp B
- Therefore:
  - Exp B = strong diagnostic baseline
  - Exp C = most paper-worthy supervision direction for the next larger test

## 9.4 GTA same-area supervision comparison (`teacher_query_limit=2000`)

This run is available at:

- `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/summary.md`

Official evaluator summary:

| Variant | Dis@1 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse baseline | 77.16 | 6.42 | 16.58 | 33.78 | 28.35% | 14.20% | 0.29% |
| Exp A current teacher | 70.27 | 8.31 | 19.20 | 39.44 | 19.40% | 14.00% | 0.32% |
| Exp B clean30 | 69.42 | 7.73 | 19.14 | 38.63 | 22.07% | 13.42% | 0.41% |
| Exp C weighted useful-angle | 63.03 | 8.74 | 21.64 | 42.17 | 17.51% | 11.56% | 0.15% |

Teacher-side training summary:

| Variant | kept / raw | removed | pair weight mode | pair weight mean |
|---|---:|---:|---|---:|
| Exp A current teacher | 2000 / 2000 | 0 | uniform | 1.0000 |
| Exp B clean30 | 1701 / 2000 | 299 | uniform | 1.0000 |
| Exp C weighted useful-angle | 2000 / 2000 | 0 | best_distance_sigmoid | 0.7572 |

Interpretation:

- GTA same-area also shows meaningful teacher noise.
- Exp B helps, so denoising matters.
- But Exp C is the current best GTA supervision line because it improves both:
  - raw `Dis@1`
  - and the more important robustness metrics
- Therefore the current GTA default should be:
  - Exp C weighted useful-angle set supervision
- If only one GTA supervision recipe is carried to larger teacher subsets or
  cross-area later, carry:
  - Exp C

## 9.5 GTA same-area Exp C follow-up (`teacher_query_limit=2000`)

This run is available at:

- `Game4Loc/work_dir/gta_exp_c_followup_runs/exp_c_followup_samearea_q2000_20260410/summary.md`

Official evaluator summary:

| Variant | Dis@1 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp C baseline (`delta=5`, `center=30`) | 63.03 | 8.74 | 21.64 | 42.17 | 17.51% | 11.56% | 0.15% |
| Exp C1 (`useful_delta_m=3`) | 63.82 | 8.74 | 20.94 | 42.23 | 16.67% | 12.69% | 0.17% |
| Exp C2 (`pair_weight_center_m=20`) | 62.72 | 8.31 | 21.49 | 41.94 | 19.05% | 10.75% | 0.23% |

Teacher-side summary:

| Variant | useful-set mean | pair-weight mean | note |
|---|---:|---:|---|
| Exp C baseline (`delta=5`, `center=30`) | 3.7090 | 0.7572 | current default |
| Exp C1 (`useful_delta_m=3`) | 2.8675 | 0.7572 | useful set became tighter |
| Exp C2 (`pair_weight_center_m=20`) | 3.7090 | 0.6141 | weighting became stricter |

Interpretation:

- Exp C appears more sensitive to useful-angle set definition than to pair
  weighting.
- `useful_delta_m=3` is **not** better than `5` for GTA same-area.
- `pair_weight_center_m=20` is **not** a more stable replacement for `30`.
- Therefore keep the current GTA default as:
  - `useful_delta_m = 5`
  - `pair_weight_center_m = 30`
- If GTA work continues later, expand teacher subset before continuing
  micro-tuning on these two knobs.


# 10. Key Metrics That Must Be Reported

For official fine-localization experiments, always report:

- `Recall@1`, `Recall@5`, `Recall@10`, `AP` if the log already provides them
- `Dis@1`, `Dis@3`, `Dis@5`
- `MA@3m`, `MA@5m`, `MA@10m`, `MA@20m`
- `worse-than-coarse count / ratio`
- `fallback count / ratio`
- `identity-H fallback count`
- `out-of-bounds count`
- `projection-invalid count`
- mean retained matches
- mean inliers
- mean inlier ratio
- mean VOP forward time / query
- mean matcher time / query
- mean total fine-localization time / query

For cached mechanism experiments, report:

- top-k oracle-best coverage
- useful-angle coverage
- useful-angle set recall
- covered-vs-missed final error


# 11. Checkpoints And Artifacts

## 11.1 Retrieval / backbone checkpoint

Use this for current comparable experiments:

- `Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`

For GTA-UAV comparable experiments, use:

- same-area:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`
- cross-area:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_cross_area.pth`

Do not use for new matched runs unless explicitly necessary:

- `Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_best.pth`
  - older / incomplete training state

## 11.2 Orientation checkpoints

- Current frozen baseline:
  - `Game4Loc/work_dir/vop/vop_0407_full_rankce_e6.pth`
- Hard clean-pair baseline:
  - `Game4Loc/work_dir/vop/vop_0409_clean30_rankce_e6.pth`
- Weighted useful-angle baseline:
  - `Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`

## 11.3 Teacher cache

- `Game4Loc/work_dir/vop/teacher_0407_full.pt`

## 11.4 Confidence verifier artifact

This exists, but is **ablation only**:

- `Game4Loc/work_dir/confidence/linear_verifier_same_area_prior_topk4.pth`

## 11.5 Cached mechanism files

Current useful cached files:

- `Game4Loc/work_dir/vop/topk_analysis/posterior_k2.json`
- `Game4Loc/work_dir/vop/topk_analysis/posterior_k4.json`
- `Game4Loc/work_dir/vop/topk_analysis/uniform_k4.json`
- `Game4Loc/work_dir/vop/supervision_diag_0409/cache_clean30.json`
- `Game4Loc/work_dir/vop/supervision_diag_0409/cache_useful5_weight30.json`
- `Game4Loc/work_dir/vop/supervision_diag_0409/clean30_k2.json`
- `Game4Loc/work_dir/vop/supervision_diag_0409/clean30_k4.json`
- `Game4Loc/work_dir/vop/supervision_diag_0409/useful5_weight30_k2.json`
- `Game4Loc/work_dir/vop/supervision_diag_0409/useful5_weight30_k4.json`

## 11.6 GTA-UAV smoke artifacts

These exist only as linkage checks:

- `Game4Loc/work_dir/vop/gta_samearea_smoke_teacher.pt`
- `Game4Loc/work_dir/vop/gta_samearea_smoke_vop.pth`

Do **not** use them for claims.


# 12. File Map For The Current Fine-Localization Line

Core evaluator and matcher path:

- `Game4Loc/eval_visloc.py`
- `Game4Loc/game4loc/evaluate/visloc.py`
- `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
- `Game4Loc/game4loc/matcher/gim_dkm.py`
- `Game4Loc/game4loc/dataset/visloc.py`
- `Game4Loc/game4loc/models/model.py`

Orientation / training utilities:

- `Game4Loc/game4loc/orientation/vop.py`
- `Game4Loc/build_vop_teacher.py`
- `Game4Loc/train_vop.py`
- `Game4Loc/build_topk_cache.py`
- `Game4Loc/eval_topk_cached.py`
- `Game4Loc/analyze_topk_hypotheses.py`
- `Game4Loc/analyze_vop.py`
- `Game4Loc/train_confidence_verifier.py`

GTA-UAV evaluation path:

- `Game4Loc/eval_gta.py`
- `Game4Loc/game4loc/evaluate/gta.py`
- `Game4Loc/game4loc/dataset/gta.py`


# 13. Reproducible Commands

All commands below assume:

- cwd: `/home/lcy/Workplace/GTA-UAV/Game4Loc`
- env: `gtauav`

## 13.1 Official evaluator baselines on 03/04 dev protocol

### rotate=90

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --test_mode pos \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --with_match --sparse --ignore_yaw --rotate 90 --num_workers 0
```

### current frozen proposer, `prior_topk=4`

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --test_mode pos \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --with_match --sparse --ignore_yaw \
  --orientation_checkpoint ./work_dir/vop/vop_0407_full_rankce_e6.pth \
  --orientation_mode prior_topk --orientation_topk 4 \
  --num_workers 0
```

### current frozen proposer, `prior_topk=2`

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --test_mode pos \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --with_match --sparse --ignore_yaw \
  --orientation_checkpoint ./work_dir/vop/vop_0407_full_rankce_e6.pth \
  --orientation_mode prior_topk --orientation_topk 2 \
  --num_workers 0
```

## 13.2 Train supervision diagnosis variants

### Exp B: hard clean-pair filter baseline

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python train_vop.py \
  --teacher_cache ./work_dir/vop/teacher_0407_full.pt \
  --output_path ./work_dir/vop/vop_0409_clean30_rankce_e6.pth \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --batch_size 16 --num_workers 0 --epochs 6 \
  --filter_best_distance_max 30
```

### Exp C: pair-confidence-weighted useful-angle set supervision

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python train_vop.py \
  --teacher_cache ./work_dir/vop/teacher_0407_full.pt \
  --output_path ./work_dir/vop/vop_0409_useful5_weight30_e6.pth \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --batch_size 16 --num_workers 0 --epochs 6 \
  --supervision_mode useful_bce \
  --useful_delta_m 5 \
  --ce_weight 1.0 \
  --pair_weight_mode best_distance_sigmoid \
  --pair_weight_center_m 30 \
  --pair_weight_scale_m 10
```

## 13.3 Official evaluator for supervision diagnosis checkpoints

### clean30, `prior_topk=4`

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --test_mode pos \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --with_match --sparse --ignore_yaw \
  --orientation_checkpoint ./work_dir/vop/vop_0409_clean30_rankce_e6.pth \
  --orientation_mode prior_topk --orientation_topk 4 \
  --num_workers 0
```

### useful-weighted, `prior_topk=4`

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --test_mode pos \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --with_match --sparse --ignore_yaw \
  --orientation_checkpoint ./work_dir/vop/vop_0409_useful5_weight30_e6.pth \
  --orientation_mode prior_topk --orientation_topk 4 \
  --num_workers 0
```

## 13.4 Cached mechanism analysis

### build cache

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python build_topk_cache.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --orientation_checkpoint ./work_dir/vop/vop_0409_useful5_weight30_e6.pth \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth \
  --num_workers 0 \
  --output_path ./work_dir/vop/supervision_diag_0409/cache_useful5_weight30.json
```

### evaluate cached top-k posterior

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python eval_topk_cached.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --cache_path ./work_dir/vop/supervision_diag_0409/cache_useful5_weight30.json \
  --strategy posterior \
  --topk 4 \
  --output_path ./work_dir/vop/supervision_diag_0409/useful5_weight30_k4.json
```

## 13.5 GTA-UAV VOP migration

Important runtime note:

- prefer:
  - `--num_workers 0`
- prefer:
  - `--with_match --sparse`

Current GTA default training recipe:

- Exp C weighted useful-angle set supervision

### build GTA same-area teacher cache

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python build_vop_teacher.py \
  --dataset gta \
  --data_root ./data/GTA-UAV-data \
  --pairs_meta_file same-area-drone2sate-train.json \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --output_path ./work_dir/vop/gta_samearea_teacher.pt
```

### train GTA same-area weighted useful-angle VOP

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python train_vop.py \
  --teacher_cache ./work_dir/vop/gta_samearea_teacher.pt \
  --output_path ./work_dir/vop/gta_samearea_useful5_weight30_e6.pth \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --batch_size 16 --num_workers 0 --epochs 6 \
  --supervision_mode useful_bce \
  --useful_delta_m 5 \
  --ce_weight 1.0 \
  --pair_weight_mode best_distance_sigmoid \
  --pair_weight_center_m 30 \
  --pair_weight_scale_m 10
```

Current one-click GTA same-area pipeline default:

```bash
./scripts/run_gta_same_area_vop_pipeline.sh
```

### GTA same-area sparse baseline without VOP

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_gta.py \
  --data_root ./data/GTA-UAV-data \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --with_match --sparse --num_workers 0
```

### GTA same-area sparse `prior_topk=4`

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_gta.py \
  --data_root ./data/GTA-UAV-data \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --with_match --sparse --num_workers 0 --batch_size 32 --gpu_ids 0 \
  --orientation_checkpoint ./work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/artifacts/exp_c_useful5_weight30_e6.pth \
  --orientation_mode prior_topk --orientation_topk 4
```

Current behavior note:

- this command writes the standard evaluation log under:
  - `Game4Loc/Log/...`
- it does **not** write per-query match files by default

### GTA cross-area switch

To move the same pipeline to cross-area, switch both:

- checkpoint:
  - `./pretrained/gta/vit_base_eva_gta_cross_area.pth`
- meta files:
  - `cross-area-drone2sate-train.json`
  - `cross-area-drone2sate-test.json`


# 14. What The Next Agent Should Say In The Paper

Preferred paper framing:

1. **Problem diagnosis**
   - post-retrieval fine localization is fragile because orientation ambiguity
     interacts with unreliable correspondences and unstable geometry

2. **Method principle**
   - predicting a small set of useful angle hypotheses is more appropriate than
     forcing a single best angle

3. **Structured solution**
   - useful-angle proposal
   - then geometry verification

4. **Supervision refinement**
   - noisy teacher signals matter
   - supervision should move from single-angle error distillation toward
     useful-angle set supervision with pair confidence

Do **not** frame the current work as:

> "we tried many heuristics and found one that works"


# 15. What The Next Agent Must Not Claim

Do **not** claim:

- "SOTA on UAV-VisLoc"
- "matched protocol win over the original paper"
- "full-protocol superiority"

unless the new experiments truly reproduce the matched reference protocol.

Current 03/04 numbers are not enough for that claim.


# 16. Immediate Recommended Next Step

If the next agent must choose one rigorous next action, it should be:

> take the **pair-confidence-weighted useful-angle set supervision** line to a
> larger strict-pos UAV-VisLoc protocol and compare it against:
> - current frozen top-k baseline
> - hard clean-pair diagnostic baseline

Do **not** reopen:

- single-angle VOP rescue
- teacher-only soft posterior distillation
- partial unfreezing
- large heuristic sweeps

unless the user explicitly requests it.


# 17. Required Output Format For Future Experiment Summaries

Every completed experiment summary must use:

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


# 18. Final Principle

This repository should not become a museum of angle sweeps and rescue logic.

Prefer:

- a cleaner method with a stronger research story

over:

- a slightly better but messier patch

unless the evidence overwhelmingly says otherwise.
