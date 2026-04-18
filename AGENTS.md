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
  - `Game4Loc/eval_gta.py`
  - `Game4Loc/eval_visloc.py`
  - `Game4Loc/game4loc/evaluate/gta.py`
  - `Game4Loc/game4loc/matcher/gim_dkm.py`
  - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
- Those uncommitted additions now include:
  - official-evaluator LoFTR baseline support
  - denser SuperPoint sparse-default refresh
  - refreshed GTA / Paper7 sparse main-table rows under the new sparse default
  - updated research handoff notes
  - a paper-writing guide focused on the current method line
- Recent project capabilities already include:
  - supervision-diagnosis support in `train_vop.py`
  - GTA-UAV teacher-cache support in `build_vop_teacher.py`
  - GTA-UAV sparse VOP evaluation support in `eval_gta.py`
  - GTA-UAV sparse VOP inference integration in `game4loc/evaluate/gta.py`
  - official evaluator LoFTR support in `eval_visloc.py`, `eval_gta.py`,
    and `game4loc/matcher/gim_dkm.py`
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
   - Later full-teacher same-area Exp C rerun:
     - run summary:
       - `Game4Loc/work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/summary.md`
     - full-teacher scope:
       - `teacher_query_limit = 0`
       - `effective_teacher_queries = 13851`
     - produced checkpoint:
       - `Game4Loc/work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/artifacts/gta_samearea_useful5_weight30_e6.pth`
     - evaluation logs:
       - same-run sparse baseline:
         - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260418_0120.log`
       - same-run sparse + VOP:
         - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260418_0139.log`
     - main result:
       - same-run sparse baseline:
         - `Dis@1 = 73.01m`
         - `MA@20 = 37.87%`
         - `fallback = 13.16%`
         - `mean_total_time = 0.2528s/query`
       - full-teacher sparse + VOP:
         - `Dis@1 = 57.63m`
         - `MA@20 = 45.92%`
         - `fallback = 9.53%`
         - `mean_total_time = 0.2652s/query`
       - historical q2000 sparse + VOP reference:
         - `Dis@1 = 62.59m`
         - `MA@20 = 43.54%`
         - `fallback = 12.02%`
         - `mean_total_time = 0.3044s/query`
     - Practical conclusion:
       - expanding GTA teacher supervision from `2000` queries to the full
         same-area teacher improves over both:
         - the same-run sparse baseline
         - the historical q2000 VOP row
       - use:
         - `Game4Loc/work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/artifacts/gta_samearea_useful5_weight30_e6.pth`
         - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260418_0139.log`
         as the newest GTA sparse + VOP reference
       - keep the q2000 run as:
         - a historical fixed-subset ablation
         - **not** the newest GTA sparse + VOP mainline row

10. **Current GTA `with_match` output behavior is log-only by default.**
    - `eval_gta.py --with_match --sparse` writes the standard app log under:
      - `Game4Loc/Log/...`
    - It does **not** dump per-query match visualization files by default.
    - Reason:
      - `GimDKM(..., sparse_save_final_vis=False)`
      - `SparseSpLgMatcher(..., save_final_matches=False)`
    - There is currently no CLI flag in `eval_gta.py` to expose this behavior.

11. **Current paper-facing main-table comparison is matcher-level, not retrieval-level.**
    - Keep retrieval fixed.
    - The paper-facing comparison should focus on:
      - dense DKM fine localization
      - sparse fine localization
      - sparse + rotate90 baseline
      - sparse + VOP
    - Supplementary external matcher rows such as:
      - LoFTR
      may be reported in appendix / reviewer-response style comparisons, but
      should not replace the four core rows above.

12. **Current paper-facing sparse baseline definition includes multi-scale by default.**
    - Do **not** split the paper table into:
      - `sparse`
      - `sparse + multi-scale`
    - Treat current sparse matching as:
      - sparse matching with its default multi-scale configuration
    - Current code status:
      - VisLoc sparse explicitly exposes multi-scale CLI controls and defaults to
        enabled
      - GTA sparse currently uses the matcher's default multi-scale path
    - Current code-level sparse matcher defaults now also use:
      - `SuperPoint detection_threshold = 0.0003`
      - `SuperPoint max_num_keypoints = 4096`
    - Keep unchanged by default:
      - current LightGlue profile
      - current cross-scale dedup setting (`0`)
      - current multi-scale policy

13. **Current paper-facing rotate baseline must be explicit.**
    - Use:
      - sparse matching
      - rotate90 search
      - candidate selection by `inlier count`
      - tie-break by `inlier ratio`
    - Do **not** describe this row with vague names like:
      - `sparse + rotate90`
    - Do **not** mix in extra heuristics for this baseline.

14. **Current UAV-VisLoc paper-facing protocol is `same-area-paper7`, not expanded strict-pos.**
    - Use:
      - `data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-train.json`
      - `data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-test.json`
      - `data/UAV_VisLoc_dataset/same-area-paper7-split-summary.json`
    - Current matched retrieval checkpoint for this protocol is:
      - `Game4Loc/work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth`
    - Current `same-area-paper7` split summary:
      - train pos queries: `1542`
      - test pos queries: `383`
    - Do **not** use expanded `pos_semipos` as the paper main table.

15. **Current GTA-UAV paper-facing main table is complete.**
   - Completed full same-area rows:
     - dense DKM
     - sparse baseline
     - sparse + rotate90 + inlier-count selection
     - sparse + VOP
   - Current matched-table logs / summaries under the denser-SP sparse default:
     - dense DKM:
       - `Game4Loc/work_dir/gta_samearea_dense_shards_20260412/merged_summary.md`
     - sparse:
       - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2000.log`
     - sparse + rotate90 + inlier-count selection:
       - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2008.log`
     - sparse + VOP:
       - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2028.log`
   - Current matched-table key result:
      - dense DKM: `Dis@1 = 50.11m`, `MA@20 = 54.81%`, `4.0410s/query`
      - sparse: `Dis@1 = 108.47m`, `MA@20 = 14.61%`, `0.0651s/query`
      - rotate90 baseline: `Dis@1 = 77.50m`, `MA@20 = 36.45%`, `0.2831s/query`
      - sparse + VOP: `Dis@1 = 62.59m`, `MA@20 = 43.54%`, `0.3044s/query`
   - Later full-teacher sparse-side follow-up:
     - run summary:
       - `Game4Loc/work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519/summary.md`
     - refreshed sparse-side logs:
       - sparse baseline rerun:
         - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260418_0120.log`
       - sparse + VOP full-teacher rerun:
         - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260418_0139.log`
     - follow-up result:
       - sparse baseline rerun:
         - `Dis@1 = 73.01m`
         - `MA@20 = 37.87%`
         - `fallback = 13.16%`
         - `mean_total_time = 0.2528s/query`
       - sparse + VOP full-teacher rerun:
         - `Dis@1 = 57.63m`
         - `MA@20 = 45.92%`
         - `fallback = 9.53%`
         - `mean_total_time = 0.2652s/query`
       - delta vs same-run sparse baseline:
         - `Dis@1`: `-15.39m`
         - `MA@20`: `+8.05pp`
         - `fallback`: `-3.63pp`
       - delta vs historical q2000 sparse + VOP row:
         - `Dis@1`: `-4.97m`
         - `MA@20`: `+2.38pp`
         - `fallback`: `-2.49pp`
   - Important scope note:
     - this later full-teacher follow-up refreshed:
       - sparse baseline
       - sparse + VOP
     - it did **not** refresh:
       - dense DKM
       - sparse + rotate90 + inlier-count selection
     - therefore do **not** silently mix the `20260418` sparse-side rerun with
       the older `20260411/20260412` four-row table as if all rows were
       refreshed under one identical run snapshot
   - Execution note:
      - the dense row was completed with a sharded runner because the evaluator
        is very slow on this machine for full same-area GTA
      - the merged markdown summary above is the formal source of record for the
        dense row
   - Practical reading:
      - GTA same-area now supports:
        - a matched four-row dense / sparse / rotate / VOP table
        - a later full-teacher sparse-side rerun
      - `sparse + VOP` remains the strongest sparse row
      - the newest sparse + VOP reference is now the full-teacher rerun at:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260418_0139.log`
      - but dense DKM is still better than the sparse line in headline
        accuracy / robustness

16. **Current UAV-VisLoc `same-area-paper7` paper-facing main table is complete.**
    - Teacher cache build completed:
      - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/teacher_samearea_paper7.pt`
    - Paper7 VOP training completed:
      - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/vop_samearea_paper7_useful5_weight30_e6.pth`
    - Completed full main-table rows:
      - dense DKM
      - sparse baseline
      - sparse + rotate90 + inlier-count selection
      - sparse + VOP
    - Current refreshed sparse-side logs under the denser-SP sparse default:
      - sparse:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2053.log`
      - sparse + rotate90 + inlier-count selection:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2058.log`
      - sparse + VOP:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2103.log`
    - Current key result:
      - dense: `Dis@1 = 241.40m`, `MA@20 = 35.51%`, `4.1128s/query`
      - sparse: `Dis@1 = 274.68m`, `MA@20 = 10.44%`, `0.0747s/query`
      - rotate90 baseline: `Dis@1 = 258.18m`, `MA@20 = 18.28%`, `0.3111s/query`
      - sparse + VOP: `Dis@1 = 257.94m`, `MA@20 = 25.59%`, `0.3291s/query`
    - Practical reading:
      - VOP still clearly improves over the explicit sparse baselines
      - but on current Paper7, VOP is still behind dense DKM in headline
        accuracy / robustness

17. **Official evaluators now support a supplementary LoFTR baseline.**
    - `Game4Loc/eval_visloc.py` now accepts:
      - `--loftr`
    - `Game4Loc/eval_gta.py` now accepts:
      - `--loftr`
    - `Game4Loc/game4loc/matcher/gim_dkm.py` now supports:
      - `match_mode=loftr`
    - Current implementation uses:
      - Kornia pretrained outdoor LoFTR
      - homography-only RANSAC inside the LoFTR matcher path
    - Treat this as:
      - a supplementary external baseline
      - **not** the new default matcher path

18. **A UAV-VisLoc Paper7 LoFTR baseline has completed.**
    - Run summary:
      - `Game4Loc/work_dir/loftr_baseline_runs/visloc_paper7_loftr_20260411/summary.md`
    - Main result:
      - `Dis@1 = 277.03m`
      - `MA@20 = 19.06%`
      - `fallback = 7.31%`
      - `mean_total_time = 0.6796s/query`
    - Practical interpretation:
      - LoFTR is more stable than raw sparse on Paper7
      - but it is still worse than:
        - dense DKM
        - sparse + VOP
      - therefore it does **not** replace the current paper-facing sparse mainline

19. **A GTA-UAV same-area LoFTR baseline has completed.**
    - Run summary:
      - `Game4Loc/work_dir/loftr_baseline_runs/gta_samearea_loftr_20260411/summary.md`
    - Main result:
      - `Dis@1 = 130.66m`
      - `MA@20 = 16.93%`
      - `fallback = 4.01%`
      - `worse-than-coarse = 54.98%`
      - `mean_total_time = 0.6972s/query`
    - Practical interpretation:
      - LoFTR reduces fallback strongly on GTA same-area
      - but the final localization quality is much worse than:
        - sparse + rotate90 + inlier-count
        - sparse + VOP
      - therefore it should remain:
        - a supplementary comparison row
        - **not** a default path

20. **Older sparse matcher-control experiments still do not justify changing the LightGlue / dedup / pyramid defaults.**
    - Summary files:
      - `Game4Loc/work_dir/visloc_sparse_yaw_matcher_control_runs/visloc_sparse_yaw_matcher_control_20260410_v2/summary.md`
      - `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/visloc_sparse_yaw_scale_contrib_20260411_005234/summary.md`
      - `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/dense_down_dedup3_followup_20260411_011421/summary.md`
    - Controlled setting:
      - UAV-VisLoc same-area
      - `with_match + sparse`
      - `use_yaw`
      - `no VOP`
    - Main conclusions:
      - switching to an "official default" LightGlue profile regressed badly in
        the current controlled setting
      - denser multi-scale pyramids increased retained matches / inliers and
        reduced fallback, but worsened final `Dis@1`
      - cross-scale dedup helped some dense-down variants relative to no dedup,
        but still did not beat the baseline
      - therefore do **not** change:
        - the LightGlue profile
        - the sparse multi-scale policy
        - the cross-scale dedup radius
        by default

21. **Denser SuperPoint has now been promoted to the code default sparse matcher setting.**
    - User decision:
      - set the denser SuperPoint route as the default sparse configuration
    - Current code-level default change:
      - `sp_detection_threshold = 0.0003`
      - `sp_max_num_keypoints = 4096`
    - Files touched for the default:
      - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
      - `Game4Loc/game4loc/matcher/gim_dkm.py`
      - `Game4Loc/game4loc/evaluate/gta.py`
      - `Game4Loc/eval_gta.py`
      - `Game4Loc/eval_visloc.py` log text
    - Important scope boundary:
      - this default promotion applies only to the SuperPoint detector density
        and keypoint cap
      - it does **not** imply:
        - a new LightGlue profile
        - cross-scale dedup by default
        - tighter copied geometry thresholds
    - Formal refresh conclusion:
      - on GTA same-area, the new default is mixed for plain sparse, roughly
        neutral-to-positive for the rotate baseline, and slightly positive for
        `sparse + VOP`
      - on UAV-VisLoc Paper7, the new default hurts plain sparse, helps the
        rotate baseline, and improves robustness / `MA@{3,5,10,20}` for
        `sparse + VOP` while keeping `Dis@1` nearly flat
      - therefore the practical reason to keep this default is:
        - the current paper mainline is `VOP + sparse`, not plain sparse
        - and the new default is acceptable to slightly helpful on that
          mainline across the refreshed formal runs

22. **A new 03/04 controlled follow-up has confirmed that geometry quality, not raw inlier count, is the current bottleneck for `VOP + sparse`.**
    - Newly exposed VisLoc sparse controls now include:
      - `--sparse_sp_detection_threshold`
      - `--sparse_sp_max_num_keypoints`
      - `--sparse_sp_nms_radius`
      - `--sparse_ransac_method`
      - `--sparse_secondary_on_fallback`
      - `--sparse_secondary_ransac_method`
      - `--sparse_secondary_mode`
      - `--sparse_secondary_accept_min_inliers`
      - `--sparse_secondary_accept_min_inlier_ratio`
      - `--sparse_ransac_reproj_threshold`
      - `--sparse_min_inliers`
      - `--sparse_min_inlier_ratio`
    - Files touched for this control-path extension:
      - `Game4Loc/eval_visloc.py`
      - `Game4Loc/game4loc/evaluate/visloc.py`
      - `Game4Loc/game4loc/matcher/gim_dkm.py`
      - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
    - Controlled setting:
      - UAV-VisLoc `03/04 same-area`
      - `with_match + sparse`
      - `prior_topk=4`
      - retrieval fixed to:
        - `./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`
      - proposer fixed to:
        - `./work_dir/vop/vop_0409_useful5_weight30_e6.pth`
    - Baseline under the current denser-SP default:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2358.log`
      - result:
        - `Dis@1 = 50.25m`
        - `MA@20 = 30.17%`
        - `fallback = 7.76%`
        - `mean_inliers = 51.2`
    - High-resolution upsampled sparse pyramid:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0004.log`
      - change:
        - `--sparse_allow_upsample True`
        - `--sparse_scales 0.8,1.0,1.6,2.4`
      - result:
        - `mean_inliers` increased to `70.3`
        - but `Dis@1` worsened to `60.53m`
      - interpretation:
        - more matches / inliers alone do **not** solve the current error mode
    - High-resolution pyramid with cross-scale dedup:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0008.log`
      - change:
        - same upsampled scales
        - `--sparse_cross_scale_dedup_radius 5`
      - result:
        - `Dis@1 = 63.33m`
        - `MA@20 = 25.86%`
      - interpretation:
        - copying the old-project dedup idea does **not** rescue the current
          full-tile post-retrieval setting
    - MINIMA-style LightGlue profile:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0012.log`
      - change:
        - `--sparse_lightglue_profile minima_ref`
      - result:
        - collapsed to `100% fallback`
      - interpretation:
        - the current sparse line is **not** bottlenecked by an overly lax
          LightGlue confidence policy
    - Tighter homography reprojection threshold:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0015.log`
      - change:
        - `--sparse_ransac_reproj_threshold 5`
      - result:
        - `Dis@1 = 54.13m`
        - `fallback = 49.14%`
      - interpretation:
        - the current line is **not** fixed by simply tightening the H-RANSAC
          threshold
    - `USAC_MAGSAC` follow-up:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0018.log`
      - change:
        - `--sparse_ransac_method USAC_MAGSAC`
      - result:
        - `Dis@1 = 47.99m`
        - `MA@20 = 31.03%`
        - `fallback = 27.59%`
      - interpretation:
        - this is the only tested single-factor change that improves raw
          `Dis@1`
        - but it weakens robustness too much to become the new default
    - `USAC_MAGSAC + min_inliers=10` follow-up:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0022.log`
      - result:
        - `fallback` dropped back to `2.59%`
        - but `Dis@1` regressed to `60.94m`
      - interpretation:
        - the MAGSAC gain came from stricter acceptance, not from a free lunch
    - Two-path fallback `per_candidate`:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0745.log`
      - change:
        - primary:
          - `--sparse_ransac_method USAC_MAGSAC`
        - secondary:
          - `--sparse_secondary_on_fallback True`
          - `--sparse_secondary_ransac_method RANSAC`
      - result:
        - `Dis@1 = 53.35m`
        - `MA@20 = 26.72%`
        - `fallback = 0.86%`
        - `secondary_takeover = 35.34%`
      - interpretation:
        - retrying the secondary matcher on every candidate is too invasive
        - it repairs fallback, but distorts the candidate ranking and hurts
          raw accuracy
    - Two-path fallback `final_only`:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0754.log`
      - change:
        - same primary / secondary methods as above
        - but only retry on the final geometry-selected angle
      - result:
        - `Dis@1 = 51.50m`
        - `MA@20 = 31.03%`
        - `fallback = 10.34%`
        - `secondary_takeover = 16.38%`
      - interpretation:
        - final-only retry is much healthier than per-candidate retry
        - but ungated takeover is still too permissive
    - Two-path fallback `final_only` with strong acceptance gate:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0801.log`
      - change:
        - `--sparse_secondary_accept_min_inliers 25`
        - `--sparse_secondary_accept_min_inlier_ratio 0.15`
      - result:
        - `secondary_takeover = 0`
        - `Dis@1 = 50.62m`
        - `fallback = 36.21%`
      - interpretation:
        - this gate is too strict
        - it effectively collapses back to raw primary behavior
    - Two-path fallback `final_only` with moderate acceptance gate:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0805.log`
      - change:
        - `--sparse_secondary_accept_min_inliers 20`
        - `--sparse_secondary_accept_min_inlier_ratio 0.10`
      - result:
        - `Dis@1 = 46.49m`
        - `MA@20 = 35.34%`
        - `fallback = 19.83%`
        - `secondary_takeover = 2.59%`
      - interpretation:
        - this is the current best 03/04 geometry-only follow-up on the
          `VOP + sparse` line
        - the gain appears to come from allowing only a very small number of
          stronger secondary takeovers
    - Pure `USAC_MAGSAC` control rerun:
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0808.log`
      - result:
        - `Dis@1 = 49.34m`
        - `MA@20 = 31.03%`
        - `fallback = 29.31%`
      - interpretation:
        - this matched rerun still trails the moderate-gate two-path variant
        - so the `46.49m` result is not explained away by the immediate
          MAGSAC-only rerun
    - Practical conclusion:
      - the current gap to dense is **not** explained mainly by "too few
        inliers"
      - copied high-resolution / dedup / tighter-threshold heuristics from the
        old local-registration project do not transfer cleanly to the current
        retrieval-top1 full-tile setting
      - a naive two-path retry is not enough:
        - `per_candidate` is too aggressive
        - ungated `final_only` is still too permissive
      - the current most promising 03/04 dev branch is:
        - primary:
          - `USAC_MAGSAC`
        - secondary:
          - `RANSAC`
        - mode:
          - `final_only`
        - acceptance gate:
          - `min_inliers >= 20`
          - `min_inlier_ratio >= 0.10`
      - treat this as:
        - a development `accuracy-first` branch for `03/04`
        - **not yet** a formal default for the paper protocols


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

- UAV-VisLoc:
  - `same-area-paper7`
- GTA-UAV:
  - same-area main-table comparison first
- not the old 03/04 development split
- not expanded `pos_semipos`

For UAV-VisLoc `same-area-paper7`, use:

- `data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-train.json`
- `data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-test.json`
- retrieval checkpoint:
  - `Game4Loc/work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth`

If another larger protocol is built later, make sure:

- positive label semantics stay strict
- evaluation remains directly comparable
- 03/04 results are labeled clearly as development-only
- `same-area-paper7` remains labeled as the current paper-facing VisLoc protocol

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

## 9.6 GTA-UAV same-area paper-facing main table status

The current paper-facing GTA same-area row set is:

1. dense DKM
2. sparse
3. sparse + rotate90 + inlier-count selection
4. sparse + VOP

Current completed logs:

- dense DKM:
  - `Game4Loc/work_dir/gta_samearea_dense_shards_20260412/merged_summary.md`
- sparse:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2000.log`
- sparse + rotate90 + inlier-count selection:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2008.log`
- sparse + VOP:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2028.log`

Official evaluator summary:

| Variant | Recall@1 | Recall@5 | Recall@10 | mAP | Dis@1 | Dis@3 | Dis@5 | MA@3 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds | mean inliers | mean inlier ratio | mean total time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense DKM | 91.11 | 99.39 | 99.54 | 94.81 | 50.11 | 165.20 | 216.94 | 5.72 | 13.51 | 29.74 | 54.81 | 1.57% | 12.00% | 1.51% | 4225.88 | 0.8757 | 4.0410s |
| sparse | 91.11 | 99.39 | 99.54 | 94.81 | 108.47 | 165.20 | 216.94 | 0.61 | 1.86 | 5.61 | 14.61 | 58.41% | 18.47% | 0.29% | 19.56 | 0.1374 | 0.0651s |
| sparse + rotate90 + inlier-count | 91.11 | 99.39 | 99.54 | 94.81 | 77.50 | 165.20 | 216.94 | 3.22 | 8.07 | 19.02 | 36.45 | 11.97% | 21.49% | 0.58% | 53.28 | 0.2036 | 0.2831s |
| sparse + VOP | 91.11 | 99.39 | 99.54 | 94.81 | 62.59 | 165.20 | 216.94 | 3.78 | 8.34 | 22.19 | 43.54 | 12.02% | 13.30% | 0.26% | 70.06 | 0.2912 | 0.3044s |

Interpretation:

- dense DKM is now the strongest full GTA same-area row in both:
  - absolute accuracy
  - robustness
- sparse-only is very fast, but it drops too much fine-localization accuracy and
  robustness.
- rotate90 + inlier-count is a strong baseline and must be kept explicit in the
  paper.
- under the denser-SP refresh, plain sparse did **not** become better overall;
  it mainly traded lower fallback for worse `Dis@1` and much worse
  `worse-than-coarse`.
- sparse + VOP is still better than the refreshed rotate baseline on the
  current full GTA same-area test in:
  - raw `Dis@1`
  - all `MA@{3,5,10,20}`
  - worse-than-coarse
  - inlier statistics
- versus dense DKM, sparse + VOP is:
  - worse in raw `Dis@1` by about `12.48m`
  - worse in `MA@20` by about `11.27pp`
  - worse in fallback by about `10.45pp`
  - but faster by about `13.3x`
- Runtime-wise:
  - VOP adds only about `0.021s/query` over the refreshed rotate baseline
  - but gives a meaningful quality gain

Practical conclusion:

- the current GTA main-table evidence already supports:
  - VOP > sparse
  - VOP > explicit rotate90 baseline
- and it now also supports a matched dense-vs-ours comparison:
  - dense DKM remains stronger in absolute quality
  - sparse + VOP remains much faster and is still the strongest sparse line

## 9.7 UAV-VisLoc `same-area-paper7` paper-facing main table

Current paper-facing run directory:

- `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411`

Training artifacts:

- teacher cache:
  - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/teacher_samearea_paper7.pt`
- VOP checkpoint:
  - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/vop_samearea_paper7_useful5_weight30_e6.pth`

Official evaluator logs:

- dense:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_0400.log`
- sparse:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2053.log`
- sparse + rotate90 + inlier-count selection:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2058.log`
- sparse + VOP:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2103.log`

Retrieval metrics are fixed across all four rows:

- `Recall@1 = 65.27`
- `Recall@5 = 87.99`
- `Recall@10 = 91.64`
- `AP = 75.29`

Official evaluator summary:

| Variant | Dis@1 | Dis@3 | Dis@5 | MA@3 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds | mean inliers | mean inlier ratio | mean total time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense DKM | 241.40 | 399.49 | 483.15 | 2.35 | 4.44 | 15.14 | 35.51 | 8.09% | 48.04% | 7.83% | 3150.2 | 0.6533 | 4.1128s |
| sparse | 274.68 | 399.49 | 483.15 | 0.00 | 1.31 | 4.96 | 10.44 | 43.60% | 78.59% | 1.04% | 19.2 | 0.1332 | 0.0747s |
| sparse + rotate90 + inlier-count | 258.18 | 399.49 | 483.15 | 0.78 | 1.83 | 5.22 | 18.28 | 16.97% | 62.40% | 0.78% | 34.9 | 0.1701 | 0.3111s |
| sparse + VOP | 257.94 | 399.49 | 483.15 | 1.57 | 3.13 | 8.62 | 25.59 | 15.93% | 60.57% | 0.78% | 38.2 | 0.1851 | 0.3291s |

Interpretation:

- sparse-only is much faster than dense, but loses too much robustness.
- rotate90 + inlier-count is a necessary baseline because it recovers a
  meaningful part of the sparse-only drop.
- under the denser-SP refresh, plain sparse got worse in headline accuracy, so
  the new default should **not** be sold as a generic sparse improvement.
- sparse + VOP is still better than the refreshed rotate baseline on Paper7 in:
  - raw `Dis@1`
  - `MA@3`
  - `MA@5`
  - `MA@10`
  - `MA@20`
  - fallback
  - worse-than-coarse
- Runtime-wise, sparse + VOP is essentially the same cost as the rotate
  baseline on Paper7, with about `0.018s/query` extra cost.

Practical conclusion:

- On current Paper7:
  - VOP clearly improves over sparse baselines
  - but it is still behind dense DKM in headline accuracy / robustness
- Therefore the current Paper7 evidence supports the paper claim:
  - VOP is the strongest sparse fine-localization variant among the compared
    sparse baselines
- But it does **not** yet support the stronger claim:
  - VOP already matches or beats dense DKM on Paper7

## 9.8 Supplementary external matcher baseline: LoFTR

LoFTR is now available as a supplementary official-evaluator baseline through:

- `eval_visloc.py --loftr`
- `eval_gta.py --loftr`

Current completed summary files:

- UAV-VisLoc Paper7:
  - `Game4Loc/work_dir/loftr_baseline_runs/visloc_paper7_loftr_20260411/summary.md`
- GTA-UAV same-area:
  - `Game4Loc/work_dir/loftr_baseline_runs/gta_samearea_loftr_20260411/summary.md`

Official evaluator summary:

| Dataset | Variant | Dis@1 | MA@20 | fallback | worse-than-coarse | mean total time |
|---|---|---:|---:|---:|---:|---:|
| UAV-VisLoc Paper7 | LoFTR | 277.03 | 19.06 | 7.31% | 65.54% | 0.6796s |
| GTA same-area | LoFTR | 130.66 | 16.93 | 4.01% | 54.98% | 0.6972s |

Interpretation:

- LoFTR often reduces fallback compared with raw sparse baselines.
- But on the current formal protocols it does **not** beat:
  - dense DKM
  - sparse + rotate90 + inlier-count
  - sparse + VOP
- Therefore LoFTR should currently be treated as:
  - a supplementary external baseline
  - **not** a replacement for the current sparse paper mainline

Important implementation note:

- The stable LoFTR path currently uses:
  - Kornia pretrained outdoor LoFTR
  - homography-only RANSAC inside the matcher path

## 9.9 Sparse matcher-control takeaways (VisLoc yaw-aligned, no VOP)

These runs were done only as controlled matcher analysis, not as paper main-table
protocols.

Current summary files:

- `Game4Loc/work_dir/visloc_sparse_yaw_matcher_control_runs/visloc_sparse_yaw_matcher_control_20260410_v2/summary.md`
- `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/visloc_sparse_yaw_scale_contrib_20260411_005234/summary.md`
- `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/dense_down_dedup3_followup_20260411_011421/summary.md`
- quick `dedup2 / dedup5` follow-up logs:
  - `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/dense_down_dedup_radius_quick_20260411_012113/`

Controlled quarter-subset highlights:

| Variant | Dis@1 | MA@20 | fallback | mean inliers | mean total time |
|---|---:|---:|---:|---:|---:|
| baseline | 69.38 | 6.90 | 62.07% | 16.3 | 0.1158s |
| LightGlue official-style profile | 73.88 | 3.45 | 100.00% | 0.0 | 0.0888s |
| query-only multi-scale | 92.73 | 6.90 | 20.69% | 22.0 | 0.0683s |
| gallery-only multi-scale | 71.50 | 10.34 | 62.07% | 14.8 | 0.0708s |
| baseline + dedup5 | 69.73 | 6.90 | 68.97% | 13.7 | 0.0735s |

Denser-pyramid highlights:

| Variant | Dis@1 | MA@20 | fallback | mean inliers | mean total time |
|---|---:|---:|---:|---:|---:|
| baseline | 70.87 | 6.90 | 68.97% | 15.6 | 0.0885s |
| dense_down | 85.57 | 10.34 | 6.90% | 27.7 | 0.1335s |
| dense_mix_up | 97.42 | 6.90 | 10.34% | 33.0 | 0.1285s |
| dense_down + dedup3 | 91.59 | 3.45 | 17.24% | 20.4 | 0.1155s |
| dense_down + dedup2 | 82.94 | 3.45 | 34.48% | 18.1 | 0.1189s |
| dense_down + dedup5 | 79.32 | 10.34 | 27.59% | 19.8 | 0.1192s |

Interpretation:

- More retained matches / more inliers did **not** automatically improve final
  localization distance.
- Denser pyramids often reduced fallback and raised inlier counts, but still
  worsened `Dis@1`.
- Cross-scale dedup could partially repair dense-down variants, with `5px`
  looking best among the quick radii tried, but it still did not beat the
  baseline.
- Therefore this older VisLoc yaw-aligned control study still supports only:
  - keeping the current LightGlue profile
  - keeping the current multi-scale policy
  - not enabling cross-scale dedup by default
- The later denser-SuperPoint default promotion came from the GTA / Paper7
  refresh on the current paper mainline, not from these older yaw-aligned
  no-VOP controls.

## 9.10 GTA sparse matcher-control transfer from an external SP+LG project

These runs were done only as **small controlled GTA same-area experiments** on the
current `sparse + VOP` line. They are not formal paper main-table rows.

Purpose:

- test whether a stronger SuperPoint+LightGlue recipe from another project
  transfers to the current GTA `VOP + sparse` pipeline
- especially:
  - denser SuperPoint detection
  - different LightGlue thresholding
  - tighter geometry thresholds
  - cross-scale dedup

Matched setup:

- dataset:
  - GTA-UAV `same-area`
- retrieval:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`
- proposer:
  - `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/artifacts/exp_c_useful5_weight30_e6.pth`
- mode:
  - `--with_match --sparse`
  - `--orientation_mode prior_topk`
  - `--orientation_topk 4`
- query subset:
  - `query_limit=345`
  - chosen because it tracks GTA full-run accuracy much better than the older
    `first-172` prefix

Relevant logs:

- baseline current sparse + VOP:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_1928.log`
- external-style LightGlue transfer attempt:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_1935.log`
- denser SP + dedup5 + tighter H threshold:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_1940.log`
- denser SP + dedup5 with current geometry threshold:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_1945.log`
- denser SP only with current geometry threshold:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_1949.log`

Controlled summary:

| Variant | Exact change vs baseline | Dis@1 | MA@20 | fallback | worse-than-coarse | mean inliers | mean total time |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | current GTA sparse + VOP | 62.90 | 39.42 | 21.16% | 10.14% | 51.54 | 0.2894s |
| external-style LG transfer | `SP: det=3e-4, kp=4096` + `LG=minima_ref` + `dedup5` + `H thr=8` | 104.80 | 6.09 | 100.00% | 0.00% | 0.00 | 0.3345s |
| denser SP + dedup5 + tight H | `SP: det=3e-4, kp=4096` + `dedup5` + `H thr=8` | 61.21 | 37.68 | 34.49% | 4.06% | 42.90 | 0.3021s |
| denser SP + dedup5 | `SP: det=3e-4, kp=4096` + `dedup5` | 60.39 | 41.74 | 12.46% | 12.17% | 63.54 | 0.2957s |
| denser SP only | `SP: det=3e-4, kp=4096` | 59.47 | 43.19 | 12.75% | 13.91% | 78.88 | 0.2941s |

Interpretation:

- The external project's LightGlue-style profile did **not** transfer cleanly.
  It collapsed into near-total fallback and should not be adopted.
- The main failure mode was **not** "more keypoints are bad."
  Instead, the damaging transfer was:
  - the LightGlue profile change itself
  - and the tighter geometry threshold (`H thr=8`)
- On current GTA `VOP + sparse`, denser SuperPoint is actually promising:
  - `Dis@1` improved from `62.90m` to `59.47m`
  - `MA@20` improved from `39.42%` to `43.19%`
  - fallback dropped from `21.16%` to `12.75%`
  - runtime stayed almost unchanged
- Cross-scale dedup at `5px` was **not** the key gain here.
  On this GTA subset, `denser SP only` was slightly better than
  `denser SP + dedup5`.
- The tradeoff is that denser SP also increased `worse-than-coarse` somewhat,
  so the small controlled result alone was not enough to justify a paper claim.

Current practical takeaway:

- This exact matcher-level follow-up has now been promoted and refreshed on the
  formal GTA / Paper7 protocols.
- The outcome of those larger runs is:
  - acceptable to slightly positive on the current `sparse + VOP` mainline
  - not a generic win for plain sparse
- Do **not** currently promote:
  - external LightGlue profile transfer
  - tighter homography thresholding copied from the external project
  - cross-scale dedup as a new default

## 9.11 Denser SuperPoint default refresh on the formal protocols

User-directed default refresh:

- sparse matcher default now uses:
  - `SuperPoint detection_threshold = 0.0003`
  - `SuperPoint max_num_keypoints = 4096`

Refreshed official runs:

- GTA same-area:
  - sparse:
    - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2000.log`
  - rotate90 baseline:
    - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2008.log`
  - sparse + VOP:
    - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2028.log`
- UAV-VisLoc Paper7:
  - sparse:
    - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2053.log`
  - rotate90 baseline:
    - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2058.log`
  - sparse + VOP:
    - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2103.log`

Cross-dataset reading:

- On both formal protocols, the main paper line still holds:
  - `sparse + VOP` is the strongest sparse fine-localization variant among the
    sparse rows
- The refreshed default does **not** make plain sparse a stronger paper row.
- The refreshed default is most compatible with:
  - rotate-aware sparse
  - `VOP + sparse`
- Therefore future matched experiments should treat the current sparse default
  as:
  - denser SuperPoint
  - unchanged LightGlue profile
  - unchanged cross-scale dedup
  - unchanged multi-scale policy

## 9.12 03/04 dev follow-up: gated dual-path geometry retry for `VOP + sparse`

These runs are still **development-only** and use the old `03/04 same-area`
protocol. They should not be quoted as formal paper headline results.

Purpose:

- continue the post-retrieval fine-localization diagnosis on the current
  `VOP + sparse` line
- test whether `USAC_MAGSAC` can be kept as the primary geometry estimator
  while using a conservative `RANSAC` retry only for fallback cases

Fixed setting:

- UAV-VisLoc `03/04 same-area`
- retrieval fixed to:
  - `Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`
- VOP fixed to:
  - `Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- mode:
  - `with_match + sparse + prior_topk=4`
- no yaw

Current key logs:

- baseline under the current denser-SP default:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2358.log`
- primary `USAC_MAGSAC`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0018.log`
- pure `USAC_MAGSAC` matched rerun:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0808.log`
- two-path `per_candidate`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0745.log`
- two-path `final_only`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0754.log`
- two-path `final_only` with `accept 25 / 0.15`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0801.log`
- two-path `final_only` with `accept 20 / 0.10`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0805.log`

Summary table:

| Variant | Dis@1 | MA@20 | fallback | worse-than-coarse | secondary_takeover | mean total time |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 50.25 | 30.17 | 7.76% | 40.52% | 0.00% | 0.3343s |
| `USAC_MAGSAC` | 47.99 | 31.03 | 27.59% | 48.28% | 0.00% | 0.2694s |
| `USAC_MAGSAC` rerun | 49.34 | 31.03 | 29.31% | 49.14% | 0.00% | 0.2651s |
| two-path `per_candidate` | 53.35 | 26.72 | 0.86% | 40.52% | 35.34% | 0.4331s |
| two-path `final_only` | 51.50 | 31.03 | 10.34% | 42.24% | 16.38% | 0.2888s |
| two-path `final_only` + `accept 25 / 0.15` | 50.62 | 29.31 | 36.21% | 52.59% | 0.00% | 0.2787s |
| two-path `final_only` + `accept 20 / 0.10` | 46.49 | 35.34 | 19.83% | 45.69% | 2.59% | 0.2822s |

Interpretation:

- `USAC_MAGSAC` is still the only single-factor geometry change that improves
  raw `Dis@1` over the current baseline.
- But naive fallback repair is not enough:
  - `per_candidate` retry over-corrects and hurts ranking badly
  - ungated `final_only` retry is cleaner, but still not strong enough
- The first clearly useful conservative variant is:
  - primary:
    - `USAC_MAGSAC`
  - secondary:
    - `RANSAC`
  - retry mode:
    - `final_only`
  - secondary acceptance:
    - `min_inliers >= 20`
    - `min_inlier_ratio >= 0.10`
- In the current matched rerun, this variant beats both:
  - the current denser-SP baseline
  - the pure `USAC_MAGSAC` rerun
- Therefore, if more 03/04 dev work is requested later, the most promising next
  branch is:
  - keep the formal paper defaults unchanged
  - continue local development on the gated `final_only` dual-path geometry
    retry

## 9.13 GTA same-area quick transfer check for the gated dual-path geometry retry

These runs were done only as a **small matched same-area experiment** on
GTA-UAV. They are not formal main-table rows.

Purpose:

- test whether the current best `03/04` development direction transfers to
  GTA same-area without changing retrieval or matcher internals
- setting kept fixed to the current GTA sparse mainline:
  - retrieval checkpoint:
    - `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`
  - VOP checkpoint:
    - `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/artifacts/exp_c_useful5_weight30_e6.pth`
  - `with_match + sparse + prior_topk=4`
  - same query subset:
    - `query_limit=345`

Current logs:

- baseline subset:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260412_0820.log`
- pure `USAC_MAGSAC` subset:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260412_0830.log`
- gated dual-path subset:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260412_0826.log`

Summary table:

| Variant | Dis@1 | MA@20 | fallback | worse-than-coarse | secondary_takeover | mean total time |
|---|---:|---:|---:|---:|---:|---:|
| GTA subset baseline | 58.33 | 45.22 | 10.72% | 13.04% | 0.00% | 0.2536s |
| GTA subset `USAC_MAGSAC` | 58.63 | 41.16 | 20.29% | 7.25% | 0.00% | 0.2400s |
| GTA subset gated dual-path | 57.60 | 41.16 | 22.32% | 7.25% | 0.87% | 0.2396s |

Gated dual-path variant details:

- primary:
  - `USAC_MAGSAC`
- secondary:
  - `RANSAC`
- mode:
  - `final_only`
- acceptance gate:
  - `min_inliers >= 20`
  - `min_inlier_ratio >= 0.10`

Interpretation:

- The `03/04` development direction does **not** transfer cleanly to GTA
  same-area.
- On this matched subset, the gated dual-path variant gives only a small raw
  `Dis@1` gain over the GTA baseline:
  - `58.33m -> 57.60m`
- But it clearly worsens the more important robustness / coverage indicators:
  - `MA@20`: `45.22% -> 41.16%`
  - `fallback`: `10.72% -> 22.32%`
- The GTA effect is therefore different from the `03/04` dev protocol:
  - `USAC_MAGSAC`-style geometry reduces `worse-than-coarse`
  - but it also pushes many more queries into fallback
- Practical conclusion:
  - keep this line as a **development-only GTA diagnostic**
  - do **not** promote it to the GTA sparse default
  - if GTA geometry work continues later, the next useful step is not more
    copied retry heuristics, but understanding why MAGSAC raises fallback so
    sharply on same-area GTA


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

For current UAV-VisLoc paper-facing `same-area-paper7` experiments, use:

- `Game4Loc/work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth`

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
- Current UAV-VisLoc `same-area-paper7` paper-facing checkpoint:
  - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/vop_samearea_paper7_useful5_weight30_e6.pth`
- Current GTA-UAV same-area paper-facing checkpoint:
  - `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/artifacts/exp_c_useful5_weight30_e6.pth`

## 11.3 Teacher cache

- `Game4Loc/work_dir/vop/teacher_0407_full.pt`
- Current UAV-VisLoc `same-area-paper7` teacher cache:
  - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/teacher_samearea_paper7.pt`

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

## 11.7 Supplementary matcher-baseline summaries

Current useful supplementary baseline summaries:

- UAV-VisLoc Paper7 LoFTR:
  - `Game4Loc/work_dir/loftr_baseline_runs/visloc_paper7_loftr_20260411/summary.md`
- GTA-UAV same-area LoFTR:
  - `Game4Loc/work_dir/loftr_baseline_runs/gta_samearea_loftr_20260411/summary.md`
- VisLoc sparse matcher controls:
  - `Game4Loc/work_dir/visloc_sparse_yaw_matcher_control_runs/visloc_sparse_yaw_matcher_control_20260410_v2/summary.md`
- VisLoc scale-contribution ablation:
  - `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/visloc_sparse_yaw_scale_contrib_20260411_005234/summary.md`
- VisLoc dense-down dedup follow-up:
  - `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/dense_down_dedup3_followup_20260411_011421/summary.md`
- VisLoc quick dedup-radius follow-up logs:
  - `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/dense_down_dedup_radius_quick_20260411_012113/`


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

## 13.6 Current paper-facing main-table commands

These commands are the current paper-facing comparison commands.

Keep retrieval fixed. Only change the fine-localization module / matcher path.

### GTA-UAV same-area main table

Use retrieval checkpoint:

- `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`

#### DKM dense matching (original matcher path)

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_gta.py \
  --data_root ./data/GTA-UAV-data \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --with_match --dense --num_workers 0
```

Important note:

- this is the original dense DKM path
- on this machine it is extremely slow for full same-area GTA

#### Sparse baseline (default multi-scale, no rotate)

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_gta.py \
  --data_root ./data/GTA-UAV-data \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --with_match --sparse --no_rotate --num_workers 0
```

#### Sparse + rotate90 + inlier-count selection baseline

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_gta.py \
  --data_root ./data/GTA-UAV-data \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --with_match --sparse --num_workers 0
```

Current evaluator behavior for this row:

- sparse default multi-scale is enabled
- rotate search is the default when `--no_rotate` is omitted
- candidate selection is:
  - best `inlier count`
  - tie-break by `inlier ratio`

#### Sparse + VOP (ours)

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

### UAV-VisLoc `same-area-paper7` main table

Use retrieval checkpoint:

- `Game4Loc/work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth`

#### Build teacher cache

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python build_vop_teacher.py \
  --dataset visloc \
  --data_root ./data/UAV_VisLoc_dataset \
  --pairs_meta_file same-area-paper7-drone2sate-train.json \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth \
  --output_path ./work_dir/paper7_main_table_runs/visloc_paper7/artifacts/teacher_samearea_paper7.pt
```

#### Train paper7 VOP (Exp C)

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python train_vop.py \
  --teacher_cache ./work_dir/paper7_main_table_runs/visloc_paper7/artifacts/teacher_samearea_paper7.pt \
  --output_path ./work_dir/paper7_main_table_runs/visloc_paper7/artifacts/vop_samearea_paper7_useful5_weight30_e6.pth \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth \
  --batch_size 16 --num_workers 0 --epochs 6 \
  --supervision_mode useful_bce \
  --useful_delta_m 5 \
  --ce_weight 1.0 \
  --pair_weight_mode best_distance_sigmoid \
  --pair_weight_center_m 30 \
  --pair_weight_scale_m 10
```

#### DKM dense matching

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-paper7-drone2sate-test.json \
  --test_mode pos \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth \
  --with_match --dense --ignore_yaw --num_workers 0
```

#### Sparse baseline (default multi-scale, no rotate)

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-paper7-drone2sate-test.json \
  --test_mode pos \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth \
  --with_match --sparse --ignore_yaw --num_workers 0 --sparse_save_final_vis False
```

#### Sparse + rotate90 + inlier-count selection baseline

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-paper7-drone2sate-test.json \
  --test_mode pos \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth \
  --with_match --sparse --ignore_yaw --rotate 90 \
  --num_workers 0 --sparse_save_final_vis False
```

Current evaluator behavior for this row:

- sparse default multi-scale is enabled
- leave `--sparse_angle_score_inlier_offset` unset
- candidate selection is:
  - best `inlier count`
  - tie-break by `inlier ratio`

#### Sparse + VOP (ours)

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-paper7-drone2sate-test.json \
  --test_mode pos \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth \
  --with_match --sparse --ignore_yaw \
  --orientation_checkpoint ./work_dir/paper7_main_table_runs/visloc_paper7/artifacts/vop_samearea_paper7_useful5_weight30_e6.pth \
  --orientation_mode prior_topk --orientation_topk 4 \
  --num_workers 0 --sparse_save_final_vis False
```

## 13.7 Supplementary external matcher baseline commands

These are supplementary comparison commands, not replacements for the four core
main-table rows.

### GTA-UAV same-area LoFTR baseline

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_gta.py \
  --data_root ./data/GTA-UAV-data \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --query_mode D2S \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --batch_size 32 --num_workers 0 --gpu_ids 0 \
  --with_match --loftr --no_rotate
```

### UAV-VisLoc `same-area-paper7` LoFTR baseline

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_visloc.py \
  --data_root ./data/UAV_VisLoc_dataset \
  --test_pairs_meta_file same-area-paper7-drone2sate-test.json \
  --test_mode pos \
  --query_mode D2S \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth \
  --with_match --loftr --ignore_yaw --num_workers 0
```


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
