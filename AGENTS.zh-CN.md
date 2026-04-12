# AGENTS.md

本文档是本仓库的研究交接与执行手册。

它写给将在当前会话之后继续推进论文工作的 agent。请在运行实验或修改代码之前先阅读本文件。

项目目标不是“再堆更多 trick”，而是写出一篇关于 **无人机到卫星图像地理定位中、检索之后的精细定位（post-retrieval fine localization）** 的、论证扎实的论文。


# 1. 范围

本仓库包含两个阶段：

1. **Retrieval**
2. **检索后的精细定位**

当前论文的聚焦范围严格限定为：

> **检索之后的精细定位**

除非用户明确要求做 retrieval 相关工作，否则不要修改 retrieval。

当前的工作性诊断是：

> 精细定位失败的主要原因是朝向歧义、不可靠对应关系和不稳定几何，而不是简单地“匹配点太少”。


# 2. 当前交接状态

截至本次交接：

- 当前活跃分支：`codex/vop-experiment`
- 当前存在未提交修改的文件：
  - `AGENTS.md`
  - `Paper.md`
  - `Game4Loc/eval_gta.py`
  - `Game4Loc/eval_visloc.py`
  - `Game4Loc/game4loc/evaluate/gta.py`
  - `Game4Loc/game4loc/matcher/gim_dkm.py`
  - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
- 这些未提交内容当前包括：
  - 官方 evaluator 的 LoFTR baseline 支持
  - 更稠密的 SuperPoint 稀疏默认配置刷新
  - 新 sparse 默认配置下刷新的 GTA / Paper7 稀疏主表行
  - 更新后的研究交接说明
  - 一份聚焦当前方法线的论文写作指南
- 最近已经具备的项目能力包括：
  - `train_vop.py` 中的 supervision-diagnosis 支持
  - `build_vop_teacher.py` 中的 GTA-UAV teacher-cache 支持
  - `eval_gta.py` 中的 GTA-UAV sparse VOP 评估支持
  - `game4loc/evaluate/gta.py` 中的 GTA-UAV sparse VOP 推理集成
  - `eval_visloc.py`、`eval_gta.py` 与 `game4loc/matcher/gim_dkm.py` 中的官方 evaluator LoFTR 支持
- 除非用户明确要求，否则不要回退用户或之前 agent 的工作。

当前研究状态：

1. **单角度 VOP 不是正确主线。**
   - 它不稳定。
   - 它没有提供足够有说服力的论文故事。

2. **当前主线是：**
   - **top-k 有用角度假设 + 几何验证**

3. **当前默认 proposer 配置：**
   - `prior_topk=4`

4. **但有一个重要警告：**
   - 旧的 `03/04` same-area 小协议现在只用于 **开发**。
   - **不要**把它作为正式论文结论的主 benchmark。

5. **当前在 03/04 开发协议上的 supervision diagnosis 状态：**
   - `current teacher baseline`
   - `hard clean-pair filter baseline`
   - `pair-confidence-weighted useful-angle set supervision`
   已全部比较过。

6. **当前最合适的正式下一步：**
   - 将 **useful-angle set supervision** 这条思路迁移到更大、更严格的协议上做正式验证。
   - `hard clean-pair filter` 只保留为诊断基线。

当前 GTA-UAV 迁移状态：

1. **VOP teacher 构建现已支持 GTA-UAV。**
   - `Game4Loc/build_vop_teacher.py` 现在接受：
     - `--dataset gta`
   - GTA teacher distance 在数据集的平面 `x/y` 空间中构建，使用：
     - query 的 `drone_loc_x_y`
     - 来自 `game4loc.dataset.gta.sate2loc` 的 gallery tile 几何

2. **GTA-UAV sparse 官方评估现已支持 VOP prior。**
   - `Game4Loc/eval_gta.py` 现在接受：
     - `--orientation_checkpoint`
     - `--orientation_mode {off, prior_single, prior_topk}`
     - `--orientation_topk`
     - `--num_workers`
   - `Game4Loc/game4loc/evaluate/gta.py` 现在支持：
     - sparse `prior_single`
     - sparse `prior_topk`

3. **当前 GTA-UAV 的优先主路径是 sparse 模式。**
   - 优先使用：
     - `--with_match --sparse`
   - 目前 GTA 的 VOP 集成只接到了：
     - sparse fine localization
   - **不要**把 GTA dense 模式评估当成当前 VOP 主线。

4. **一条 same-area GTA smoke test 已经端到端跑通。**
   - 覆盖内容：
     - GTA teacher cache build
     - GTA VOP training
     - GTA sparse `prior_topk=4` evaluation
   - Smoke 产物：
     - `Game4Loc/work_dir/vop/gta_samearea_smoke_teacher.pt`
     - `Game4Loc/work_dir/vop/gta_samearea_smoke_vop.pth`
   - 重要警告：
     - 这些 smoke 产物 **仅用于验证流水线接通**
     - **不要**把其指标当作论文证据

5. **当前机器上的 GTA 评估环境说明。**
   - 在这台机器上，优先使用：
     - `--num_workers 0`
   - 原因：
     - GTA eval 在多进程 dataloader 下可能报错：
       - `OSError: [Errno 95] Operation not supported`

6. **GTA-UAV evaluator 现在会输出完整的稳健性摘要。**
   - 当前 GTA 官方日志包含：
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
   - 这只是日志 / 摘要层面的补充。
   - 它**不会**改变 GTA 评估语义。

7. **一组固定子集的 GTA same-area supervision 比较已完成。**
   - 协议：
     - same-area
     - sparse
     - 全量 same-area test
     - retrieval checkpoint 固定为 GTA same-area 官方权重
     - teacher subset 固定为 `2000` 个有效 query
     - `prior_topk=4`
   - 运行摘要：
     - `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/summary.md`
   - 主要结果：
     - baseline: `Dis@1 = 77.16m`
     - Exp A current teacher: `70.27m`
     - Exp B clean30: `69.42m`
     - Exp C weighted useful-angle: `63.03m`

8. **当前 GTA 默认 VOP 训练配方是 Exp C。**
   - 使用：
     - `useful_bce`
     - `useful_delta_m = 5`
     - `ce_weight = 1.0`
     - `pair_weight_mode = best_distance_sigmoid`
     - `pair_weight_center_m = 30`
     - `pair_weight_scale_m = 10`
   - Exp A / Exp B 只保留为诊断基线。

9. **一个最小规模的 GTA same-area Exp C follow-up 已完成。**
   - 固定协议：
     - same-area
     - sparse
     - 全量 same-area test
     - teacher subset 固定为 `2000`
     - `prior_topk=4`
   - 运行摘要：
     - `Game4Loc/work_dir/gta_exp_c_followup_runs/exp_c_followup_samearea_q2000_20260410/summary.md`
   - 比较内容：
     - baseline Exp C:
       - `useful_delta_m = 5`
       - `pair_weight_center_m = 30`
     - Exp C1:
       - `useful_delta_m = 3`
     - Exp C2:
       - `pair_weight_center_m = 20`
   - 主要结果：
     - baseline Exp C 仍是当前应保留的默认设置
     - Exp C1 回退更明显
     - Exp C2 在原始 `Dis@1` 上略好，但稳健性变弱
   - 实际结论：
     - 当前 GTA same-area 线应继续保留：
       - `useful_delta_m = 5`
       - `pair_weight_center_m = 30`
     - 如果以后继续做 GTA，先扩大 teacher subset，再去做这两个 knob 的微调扫描

10. **当前 GTA `with_match` 输出默认只写日志。**
    - `eval_gta.py --with_match --sparse` 会把标准 app log 写到：
      - `Game4Loc/Log/...`
    - 默认**不会**导出逐 query 的 match 可视化文件。
    - 原因：
      - `GimDKM(..., sparse_save_final_vis=False)`
      - `SparseSpLgMatcher(..., save_final_matches=False)`
    - 目前 `eval_gta.py` 中没有暴露这一行为的 CLI flag。

11. **当前面向论文的主表比较是 matcher-level，而不是 retrieval-level。**
    - 保持 retrieval 固定。
    - 面向论文的比较应聚焦：
      - dense DKM fine localization
      - sparse fine localization
      - sparse + rotate90 baseline
      - sparse + VOP
    - 补充性的外部 matcher 行，比如：
      - LoFTR
      可以写在附录或 reviewer-response 风格的比较中，但不能替代上述四个核心行。

12. **当前面向论文的 sparse baseline 定义默认包含 multi-scale。**
    - **不要**把论文表拆成：
      - `sparse`
      - `sparse + multi-scale`
    - 当前 sparse matching 应被视为：
      - 使用其默认 multi-scale 配置的 sparse matching
    - 当前代码状态：
      - VisLoc sparse 明确暴露了 multi-scale CLI 控制，并默认开启
      - GTA sparse 当前使用 matcher 默认的 multi-scale 路径
    - 当前代码级 sparse matcher 默认值还包括：
      - `SuperPoint detection_threshold = 0.0003`
      - `SuperPoint max_num_keypoints = 4096`
    - 默认保持不变：
      - 当前 LightGlue profile
      - 当前 cross-scale dedup 设置（`0`）
      - 当前 multi-scale policy

13. **当前面向论文的 rotate baseline 必须明确写清。**
    - 使用：
      - sparse matching
      - rotate90 search
      - 按 `inlier count` 选候选
      - 并列时按 `inlier ratio` 打破
    - **不要**用像 `sparse + rotate90` 这样模糊的名字描述这一行。
    - **不要**给这个 baseline 混入额外启发式。

14. **当前 UAV-VisLoc 面向论文的协议是 `same-area-paper7`，不是 expanded strict-pos。**
    - 使用：
      - `data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-train.json`
      - `data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-test.json`
      - `data/UAV_VisLoc_dataset/same-area-paper7-split-summary.json`
    - 当前与该协议匹配的 retrieval checkpoint 为：
      - `Game4Loc/work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth`
    - 当前 `same-area-paper7` split summary：
      - train pos queries: `1542`
      - test pos queries: `383`
    - **不要**把 expanded `pos_semipos` 用作论文主表。

15. **当前 GTA-UAV 面向论文的主表已经完成。**
    - 已完成的 full same-area 行：
      - dense DKM
      - sparse baseline
      - sparse + rotate90 + inlier-count selection
      - sparse + VOP
    - 更稠密 SP sparse 默认配置下，当前刷新的日志 / 摘要：
      - dense DKM:
        - `Game4Loc/work_dir/gta_samearea_dense_shards_20260412/merged_summary.md`
      - sparse:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2000.log`
      - sparse + rotate90 + inlier-count selection:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2008.log`
      - sparse + VOP:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2028.log`
    - 当前关键结果：
      - dense DKM: `Dis@1 = 50.11m`, `MA@20 = 54.81%`, `4.0410s/query`
      - sparse: `Dis@1 = 108.47m`, `MA@20 = 14.61%`, `0.0651s/query`
      - rotate90 baseline: `Dis@1 = 77.50m`, `MA@20 = 36.45%`, `0.2831s/query`
      - sparse + VOP: `Dis@1 = 62.59m`, `MA@20 = 43.54%`, `0.3044s/query`
    - 执行说明：
      - dense 行是用分片 runner 完成的，因为在这台机器上 full same-area GTA 的 evaluator 非常慢
      - 上述 merged markdown summary 是 dense 行的正式记录来源
    - 实际解读：
      - GTA same-area 现在已经支持在 matched retrieval 下做完整的 dense / sparse / rotate / VOP 比较
      - `sparse + VOP` 仍然是最强 sparse 行
      - 但 dense DKM 在 headline accuracy 与 robustness 上仍优于 `sparse + VOP`

16. **当前 UAV-VisLoc `same-area-paper7` 面向论文的主表已经完成。**
    - Teacher cache build 已完成：
      - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/teacher_samearea_paper7.pt`
    - Paper7 VOP training 已完成：
      - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/vop_samearea_paper7_useful5_weight30_e6.pth`
    - 已完成的 full main-table 行：
      - dense DKM
      - sparse baseline
      - sparse + rotate90 + inlier-count selection
      - sparse + VOP
    - 更稠密 SP sparse 默认配置下，当前刷新的 sparse 侧日志：
      - sparse:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2053.log`
      - sparse + rotate90 + inlier-count selection:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2058.log`
      - sparse + VOP:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2103.log`
    - 当前关键结果：
      - dense: `Dis@1 = 241.40m`, `MA@20 = 35.51%`, `4.1128s/query`
      - sparse: `Dis@1 = 274.68m`, `MA@20 = 10.44%`, `0.0747s/query`
      - rotate90 baseline: `Dis@1 = 258.18m`, `MA@20 = 18.28%`, `0.3111s/query`
      - sparse + VOP: `Dis@1 = 257.94m`, `MA@20 = 25.59%`, `0.3291s/query`
    - 实际解读：
      - VOP 依然明显优于显式 sparse baselines
      - 但在当前 Paper7 上，VOP 的 headline accuracy / robustness 仍落后于 dense DKM

17. **官方 evaluators 现在支持补充性的 LoFTR baseline。**
    - `Game4Loc/eval_visloc.py` 现在接受：
      - `--loftr`
    - `Game4Loc/eval_gta.py` 现在接受：
      - `--loftr`
    - `Game4Loc/game4loc/matcher/gim_dkm.py` 现在支持：
      - `match_mode=loftr`
    - 当前实现使用：
      - Kornia pretrained outdoor LoFTR
      - LoFTR matcher 路径内部仅使用 homography-only RANSAC
    - 它应被视为：
      - 补充性的外部 baseline
      - **不是**新的默认 matcher 路径

18. **一条 UAV-VisLoc Paper7 LoFTR baseline 已完成。**
    - 运行摘要：
      - `Game4Loc/work_dir/loftr_baseline_runs/visloc_paper7_loftr_20260411/summary.md`
    - 主要结果：
      - `Dis@1 = 277.03m`
      - `MA@20 = 19.06%`
      - `fallback = 7.31%`
      - `mean_total_time = 0.6796s/query`
    - 实际解读：
      - LoFTR 在 Paper7 上比 raw sparse 更稳定
      - 但它仍然弱于：
        - dense DKM
        - sparse + VOP
      - 因此它**不能**替代当前面向论文的 sparse 主线

19. **一条 GTA-UAV same-area LoFTR baseline 已完成。**
    - 运行摘要：
      - `Game4Loc/work_dir/loftr_baseline_runs/gta_samearea_loftr_20260411/summary.md`
    - 主要结果：
      - `Dis@1 = 130.66m`
      - `MA@20 = 16.93%`
      - `fallback = 4.01%`
      - `worse-than-coarse = 54.98%`
      - `mean_total_time = 0.6972s/query`
    - 实际解读：
      - LoFTR 在 GTA same-area 上显著降低了 fallback
      - 但最终定位质量远差于：
        - sparse + rotate90 + inlier-count
        - sparse + VOP
      - 因此它应继续作为：
        - 补充比较行
        - **不是**默认路径

20. **更早期的 sparse matcher-control 实验仍不足以支持改动 LightGlue / dedup / pyramid 默认值。**
    - 摘要文件：
      - `Game4Loc/work_dir/visloc_sparse_yaw_matcher_control_runs/visloc_sparse_yaw_matcher_control_20260410_v2/summary.md`
      - `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/visloc_sparse_yaw_scale_contrib_20260411_005234/summary.md`
      - `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/dense_down_dedup3_followup_20260411_011421/summary.md`
    - 控制设置：
      - UAV-VisLoc same-area
      - `with_match + sparse`
      - `use_yaw`
      - `no VOP`
    - 主要结论：
      - 切到一种“official default”风格的 LightGlue profile 会在当前控制设置下明显退化
      - 更密的 multi-scale pyramid 会增加 retained matches / inliers 并减少 fallback，但会恶化最终 `Dis@1`
      - cross-scale dedup 对某些 dense-down 变体比无 dedup 略有帮助，但仍未超越 baseline
      - 因此默认**不要**修改：
        - LightGlue profile
        - sparse multi-scale policy
        - cross-scale dedup radius

21. **更稠密的 SuperPoint 现已提升为代码默认 sparse matcher 配置。**
    - 用户决策：
      - 将更稠密的 SuperPoint 路线设为默认 sparse 配置
    - 当前代码级默认变化：
      - `sp_detection_threshold = 0.0003`
      - `sp_max_num_keypoints = 4096`
    - 涉及文件：
      - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
      - `Game4Loc/game4loc/matcher/gim_dkm.py`
      - `Game4Loc/game4loc/evaluate/gta.py`
      - `Game4Loc/eval_gta.py`
      - `Game4Loc/eval_visloc.py` 的日志文本
    - 重要范围边界：
      - 这一默认升级只适用于 SuperPoint 检测密度与 keypoint 上限
      - 它**不意味着**：
        - 新的 LightGlue profile
        - 默认启用 cross-scale dedup
        - 复制来的更紧几何阈值
    - 正式刷新后的结论：
      - 在 GTA same-area 上，这个新默认对 plain sparse 是 mixed 的，对 rotate baseline 大致中性到略正，对 `sparse + VOP` 略有正效应
      - 在 UAV-VisLoc Paper7 上，新默认会伤害 plain sparse、帮助 rotate baseline，并在 `Dis@1` 基本不变时提升 `sparse + VOP` 的 robustness / `MA@{3,5,10,20}`
      - 因此保留这一默认的实际理由是：
        - 当前论文主线是 `VOP + sparse`，不是 plain sparse
        - 在刷新后的正式实验上，这个新默认对该主线可接受到略有帮助

22. **一个新的 03/04 控制 follow-up 已确认：`VOP + sparse` 当前瓶颈是几何质量，而不是原始 inlier 数量。**
    - 新暴露出的 VisLoc sparse controls 包括：
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
    - 为这一控制路径扩展所改动的文件：
      - `Game4Loc/eval_visloc.py`
      - `Game4Loc/game4loc/evaluate/visloc.py`
      - `Game4Loc/game4loc/matcher/gim_dkm.py`
      - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
    - 控制设置：
      - UAV-VisLoc `03/04 same-area`
      - `with_match + sparse`
      - `prior_topk=4`
      - retrieval 固定为：
        - `./pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`
      - proposer 固定为：
        - `./work_dir/vop/vop_0409_useful5_weight30_e6.pth`
    - 当前 denser-SP 默认下的 baseline：
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2358.log`
      - result:
        - `Dis@1 = 50.25m`
        - `MA@20 = 30.17%`
        - `fallback = 7.76%`
        - `mean_inliers = 51.2`
    - 高分辨率上采样 sparse pyramid：
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0004.log`
      - change:
        - `--sparse_allow_upsample True`
        - `--sparse_scales 0.8,1.0,1.6,2.4`
      - result:
        - `mean_inliers` 增加到 `70.3`
        - 但 `Dis@1` 恶化到 `60.53m`
      - interpretation:
        - 仅增加 matches / inliers **并不能**解决当前错误模式
    - 带 cross-scale dedup 的高分辨率 pyramid：
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0008.log`
      - change:
        - 相同的上采样 scales
        - `--sparse_cross_scale_dedup_radius 5`
      - result:
        - `Dis@1 = 63.33m`
        - `MA@20 = 25.86%`
      - interpretation:
        - 复制旧项目的 dedup 思路并**不能**救回当前 retrieval 后的 full-tile 设置
    - MINIMA 风格 LightGlue profile：
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0012.log`
      - change:
        - `--sparse_lightglue_profile minima_ref`
      - result:
        - 直接退化为 `100% fallback`
      - interpretation:
        - 当前 sparse 线的瓶颈**不是** LightGlue 置信策略过于宽松
    - 更严格的 homography reprojection threshold：
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0015.log`
      - change:
        - `--sparse_ransac_reproj_threshold 5`
      - result:
        - `Dis@1 = 54.13m`
        - `fallback = 49.14%`
      - interpretation:
        - 这一条线**不能**靠简单收紧 H-RANSAC 阈值来修复
    - `USAC_MAGSAC` follow-up：
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0018.log`
      - change:
        - `--sparse_ransac_method USAC_MAGSAC`
      - result:
        - `Dis@1 = 47.99m`
        - `MA@20 = 31.03%`
        - `fallback = 27.59%`
      - interpretation:
        - 这是唯一一个能提升原始 `Dis@1` 的单因素改动
        - 但它会过度削弱稳健性，因此不能成为新默认值
    - `USAC_MAGSAC + min_inliers=10` follow-up：
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0022.log`
      - result:
        - `fallback` 降回 `2.59%`
        - 但 `Dis@1` 回退到 `60.94m`
      - interpretation:
        - MAGSAC 的收益来自更严格的接受策略，而不是“白赚”
    - 双路径 fallback `per_candidate`：
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
        - 对每个 candidate 都重试 secondary matcher 过于激进
        - 它修复了 fallback，但扭曲了 candidate ranking，损害了原始精度
    - 双路径 fallback `final_only`：
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0754.log`
      - change:
        - primary / secondary 方法同上
        - 但只在最终几何选中的 angle 上做 retry
      - result:
        - `Dis@1 = 51.50m`
        - `MA@20 = 31.03%`
        - `fallback = 10.34%`
        - `secondary_takeover = 16.38%`
      - interpretation:
        - final-only retry 比 per-candidate retry 健康得多
        - 但不加门控的 takeover 仍然过于宽松
    - 带强接受门控的双路径 fallback `final_only`：
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
        - 这一门控过严
        - 实际上等于退回到原始 primary 行为
    - 带中等接受门控的双路径 fallback `final_only`：
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
        - 这是当前 `03/04` 上 `VOP + sparse` 这条线最好的 geometry-only follow-up
        - 其收益似乎来自只允许极少数、更强的 secondary takeover
    - 纯 `USAC_MAGSAC` 对照重跑：
      - log:
        - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0808.log`
      - result:
        - `Dis@1 = 49.34m`
        - `MA@20 = 31.03%`
        - `fallback = 29.31%`
      - interpretation:
        - 这次 matched rerun 仍落后于中等门控双路径变体
        - 因而 `46.49m` 的结果不能被简单解释为 MAGSAC-only 的偶然波动
    - 实际结论：
      - 当前与 dense 的差距**主要不是**“inlier 太少”导致的
      - 旧 local-registration 项目中的高分辨率 / dedup / 更紧阈值启发式，不能干净迁移到当前 retrieval-top1 的 full-tile 设置
      - 朴素双路径 retry 并不够：
        - `per_candidate` 过于激进
        - 不加门控的 `final_only` 仍过宽
      - 当前最有前景的 03/04 开发分支是：
        - primary:
          - `USAC_MAGSAC`
        - secondary:
          - `RANSAC`
        - mode:
          - `final_only`
        - acceptance gate:
          - `min_inliers >= 20`
          - `min_inlier_ratio >= 0.10`
      - 它应被视为：
        - 一个用于 `03/04` 的开发期 `accuracy-first` 分支
        - **尚未**成为论文协议的正式默认值


# 3. 硬性约束

除非用户明确改变范围，否则**不要**：

- 修改 retrieval training；
- 重新设计 retrieval backbone；
- 默默重解释原始 pose metadata（`Phi1`, `Phi2`, `Omega`, `Kappa` 等）；
- 在不匹配协议下声称 “SOTA”；
- 把 subset 数字和 full-protocol 论文数字当成可直接对比的结果；
- 无休止地叠加救火式启发式；
- 把暴力角度扫描写成论文贡献；
- 在当前 visual-orientation 主线上使用 yaw metadata / yaw 参数；
- 除非用户明确要求，否则不要重写 matcher internals。


# 4. 必须遵守的仓库事实

- UAV-VisLoc 的预处理路径是：
  - `scripts/prepare_dataset/visloc.py`
- 当前 VisLoc evaluator 只在以下条件下应用 fine localization：
  - `query_mode="D2S"`
  - 仅对 retrieval top-1 gallery
- 当前稳定的 yaw 约定：
  - 当启用 `--use_yaw` 时，会将 query UAV 图像按 `-Phi1` 旋转，以对齐北向朝上的卫星图像
- GTA-UAV 使用平面定位坐标：
  - query 位置：
    - `drone_loc_x_y`
  - gallery tile 几何：
    - `game4loc.dataset.gta.sate2loc`
- 当前主要的 VisLoc 路径：
  - `Game4Loc/eval_visloc.py`
  - `Game4Loc/game4loc/evaluate/visloc.py`
  - `Game4Loc/game4loc/dataset/visloc.py`
  - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
  - `Game4Loc/game4loc/matcher/gim_dkm.py`
  - `scripts/prepare_dataset/visloc.py`
- 当前主要的 GTA-UAV fine-localization 路径：
  - `Game4Loc/eval_gta.py`
  - `Game4Loc/game4loc/evaluate/gta.py`
  - `Game4Loc/game4loc/dataset/gta.py`
  - `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
  - `Game4Loc/game4loc/matcher/gim_dkm.py`
  - `Game4Loc/build_vop_teacher.py`
  - `Game4Loc/train_vop.py`

当前 GTA-UAV evaluator 的事实：

- fine localization 当前只应用在：
  - `query_mode="D2S"`
  - retrieval top-1 gallery only
- 当前 GTA-UAV 中的 VOP 集成是：
  - 仅 sparse
  - 仅 retrieval-top1

如果要改 orientation handling，应在 evaluator / model / matcher 层改，而不是改 metadata 语义。


# 5. 环境与可复现性

做可比较实验时，应使用相同环境：

- Conda env: `gtauav`
- 常用解释器：
  - `/home/lcy/miniconda3/envs/gtauav/bin/python`

执行规则：

- 可比较实验应从以下目录运行：
  - `/home/lcy/Workplace/GTA-UAV/Game4Loc`
- 优先：
  - `WANDB_MODE=disabled`
- 保持命令可复制粘贴复现。
- 对于要比较的运行，不要混用不同解释器。
- 在当前机器上做 GTA-UAV 评估时，优先：
  - `--num_workers 0`


# 6. 评估分类

本项目当前有 **三种不同的评估角色**。不要混用。

## A. Official evaluator

使用：

- `Game4Loc/eval_visloc.py`
- `Game4Loc/eval_gta.py`

用途：

- headline results
- 最终 `Dis@K`
- 阈值成功率
- fallback / worse-than-coarse / runtime

这是唯一可用于正式 headline table 的结果来源。

数据集映射：

- UAV-VisLoc:
  - `Game4Loc/eval_visloc.py`
- GTA-UAV:
  - `Game4Loc/eval_gta.py`

## B. Cached evaluator

使用：

- `Game4Loc/build_topk_cache.py`
- `Game4Loc/eval_topk_cached.py`

用途：

- 仅用于机制分析
- oracle-best coverage
- useful-angle coverage
- covered-vs-missed error analysis

**不要**把 cached 数字当作论文 headline results。

当前限制：

- cached top-k analysis 当前只接到了：
  - UAV-VisLoc
- 它**尚未**移植到：
  - GTA-UAV

## C. Training supervision diagnosis

使用：

- `Game4Loc/train_vop.py`
- `Game4Loc/build_vop_teacher.py`

用途：

- 理解监督是否有噪声
- 比较 teacher 变体
- 比较 useful-angle supervision scheme

这属于开发分析，而不是最终 benchmark 报告。

当前支持状态：

- `build_vop_teacher.py` 现在支持：
  - UAV-VisLoc
  - GTA-UAV
- `train_vop.py` 仍然是 teacher-cache 驱动，并在两个数据集之间共享


# 7. 数据集协议及其含义

## 7.1 03/04 same-area 小协议

这是基于 UAV-VisLoc 的 `03` 和 `04` 区域构建的旧协议。

它只能用作：

- **开发协议**
- **监督诊断协议**

**不要**把它作为论文主 benchmark。

重要事实：

- 划分生成绑定在 `scripts/prepare_dataset/visloc.py`
- same-area train/test 只由 `03` 和 `04` 区域构成
- 当前使用的本地文件：
  - `data/UAV_VisLoc_dataset/same-area-drone2sate-train.json`
  - `data/UAV_VisLoc_dataset/same-area-drone2sate-test.json`
- 当前官方开发评估通常使用：
  - `--test_mode pos`

在 `--test_mode pos` 下：

- JSON 中 query 总数：`302`
- 实际参与 fine-localization 评估的 query：`116`
- 当前 evaluator 中的 gallery 大小：`17528`

为什么只有 `116` 个 query？

- evaluator 只保留 `pair_pos_sate_img_list` 非空的 query

## 7.2 扩展 `pos_semipos`

它曾被作为一次 stress test 测试过。

重要结论：

- 它显著增加 query 数量
- 但它也改变了 positive-label 语义
- 因此它**不是** `strict-pos` 的干净替代

那次 stress test 的具体结果：

- 协议：same-area，`test_mode=pos_semipos`
- query 数从 `116` 增加到 `302`
- 但指标语义发生了变化
- 在这个更宽松的协议下：
  - `rotate=90`: 大约 `90.68m`
  - `prior_topk=4`: 大约 `101.17m`

解释：

- 这不是 top-k 路线本身失败的证据
- 它说明：如果没有明确讨论，`pos_semipos` 不应替代 strict-pos 成为主 benchmark

除非用户明确想使用更宽松协议，否则不要把 `pos_semipos` 当作正式 benchmark。

## 7.3 正式 benchmark 目标

下一个 agent 应把真正面向论文的 benchmark 视为：

- UAV-VisLoc:
  - `same-area-paper7`
- GTA-UAV:
  - 先做 same-area main-table comparison
- 而不是旧的 03/04 开发划分
- 也不是扩展后的 `pos_semipos`

对于 UAV-VisLoc `same-area-paper7`，使用：

- `data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-train.json`
- `data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-test.json`
- retrieval checkpoint:
  - `Game4Loc/work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth`

如果以后构建了另一个更大协议，务必保证：

- positive label semantics 仍然严格
- evaluation 仍然直接可比
- 03/04 结果明确标注为仅用于开发
- `same-area-paper7` 仍明确标注为当前面向论文的 VisLoc 协议

## 7.4 GTA-UAV 协议状态

当前本地 GTA-UAV 协议文件：

- `data/GTA-UAV-data/same-area-drone2sate-train.json`
- `data/GTA-UAV-data/same-area-drone2sate-test.json`
- `data/GTA-UAV-data/cross-area-drone2sate-train.json`
- `data/GTA-UAV-data/cross-area-drone2sate-test.json`

匹配的 retrieval checkpoints：

- same-area:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`
- cross-area:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_cross_area.pth`

当前迁移建议：

- 先在 same-area 上做：
  - pipeline shakeout
  - supervision debugging
  - command validation
- 然后再分别比较 matched 的 GTA 设置：
  - same-area 配 same-area checkpoint
  - cross-area 配 cross-area checkpoint

**不要**：

- 将 same-area JSON 与 cross-area checkpoint 混用
- 将 cross-area JSON 与 same-area checkpoint 混用
- 把 smoke-test 的 GTA 指标拿来当证据

当前 smoke 状态：

- `query_limit=2` 的 same-area smoke 已经端到端跑通
- 这证明 GTA VOP 路径已经接通
- 但**不代表**方法质量已经被证明


# 8. 当前方法线

## 8.1 被拒绝 / 降级的方法线

### 单角度 VOP

状态：

- **已否决，不作为论文主线**

原因：

- 不稳定
- 太接近“猜一个角度”
- 与观察到的 angle-error surface 不匹配

### Confidence-aware verification

状态：

- **目前仅作为 ablation**

原因：

- 测试过一个轻量 verifier
- 它在一次运行中带来了轻微平均提升
- 但恶化了 `worse-than-coarse` / fallback 行为
- 还不够干净，暂时不能成为默认第二模块

## 8.2 当前主线

当前方法原则：

> 预测一小组有用角度假设，然后让几何验证在它们之间做决定

当前代码默认值：

- `orientation_mode=prior_topk`
- `orientation_topk=4`

当前 proposer checkpoint：

- `Game4Loc/work_dir/vop/vop_0407_full_rankce_e6.pth`

重要解释：

- proposer 目前还不足以被包装成一个完美的 orientation estimator
- 更合适的理解是：它是一个 **useful-angle proposer**


# 9. 当前实验结论

这些是最重要的结论。除非有充分理由，下一个 agent 不应从零重新推导它们。

## 9.1 在 03/04 开发协议上，top-k 优于 single-angle

Official evaluator 已经显示：

- `rotate=90`: `Dis@1 = 59.47m`, `MA@20 = 24.14%`
- 当前冻结 VOP 的 `prior_topk=2`: `Dis@1 = 58.55m`, `MA@20 = 29.31%`
- 当前冻结 VOP 的 `prior_topk=4`: `Dis@1 = 49.14m`, `MA@20 = 31.03%`

因此：

- top-k useful-angle proposal 优于简单的四角度 baseline
- 但这仍然只是 **开发协议证据**

## 9.2 Angle-error surface 往往不是单一尖峰

Cached analysis 显示，很多样本具有：

- useful-angle 区间
- 多个 useful mode
- 多于一个可接受角度

因此：

- 强行预测单个 best angle 在结构上就是不匹配的

## 9.3 当前 03/04 开发协议上的 supervision diagnosis

共比较了三种 supervision 变体：

### A. Current teacher baseline

- checkpoint: `Game4Loc/work_dir/vop/vop_0407_full_rankce_e6.pth`

### B. Hard clean-pair filter baseline

- checkpoint: `Game4Loc/work_dir/vop/vop_0409_clean30_rankce_e6.pth`
- filter: 只保留 `best localization error < 30m` 的训练对

观察解释：

- 它提升了开发协议上的性能
- 证明监督噪声确实重要
- 但它仍是一个粗糙的阈值型 baseline
- 很可能偏向简单 pair

### C. Pair-confidence-weighted useful-angle set supervision

- checkpoint: `Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- useful set:
  - `{theta | error(theta) <= best + 5m}`
- pair weight:
  - 基于 `best_distance` 的 sigmoid

观察解释：

- 它与问题结构更一致
- 但当前最小实现学到了一个过于平坦的 posterior
- 仍然是更值得写入论文的方向

### Supervision diagnosis 的实际结论

如果只把一种 supervision 思路迁移到更大的 strict-pos 协议：

- **选择 Exp C**：
  - pair-confidence-weighted useful-angle set supervision

而 Exp B 只保留为：

- 诊断基线
- teacher noise 确实重要的证据

### 最有用的开发期结果表

Official evaluator：

| 变体 | top-k | Dis@1 | MA@5 | MA@10 | MA@20 | worse-than-coarse | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|
| current teacher | 2 | 58.55 | 0.86 | 5.17 | 29.31 | 53.45% | 19.83% |
| current teacher | 4 | 49.14 | 2.59 | 6.03 | 31.03 | 43.97% | 18.10% |
| hard clean-pair `<30m` | 2 | 48.64 | 1.72 | 5.17 | 29.31 | 52.59% | 32.76% |
| hard clean-pair `<30m` | 4 | 45.51 | 4.31 | 6.90 | 30.17 | 44.83% | 23.28% |
| weighted useful-angle | 2 | 60.48 | 3.45 | 9.48 | 27.59 | 56.03% | 25.00% |
| weighted useful-angle | 4 | 48.86 | 2.59 | 8.62 | 37.93 | 40.52% | 11.21% |

Cached mechanism view：

| 变体 | top-k | Oracle-best cov | Useful@+5m cov | Useful@+5m recall |
|---|---:|---:|---:|---:|
| current teacher | 2 | 0.1379 | 0.3103 | 0.1024 |
| current teacher | 4 | 0.1897 | 0.4397 | 0.2112 |
| hard clean-pair `<30m` | 2 | 0.1121 | 0.3103 | 0.1226 |
| hard clean-pair `<30m` | 4 | 0.2328 | 0.4569 | 0.2532 |
| weighted useful-angle | 2 | 0.0862 | 0.2845 | 0.0874 |
| weighted useful-angle | 4 | 0.1638 | 0.4138 | 0.1745 |

这张表的解释：

- Exp B 当前给出了最佳原始开发协议结果，但它仍是粗糙的 hard-filter baseline
- Exp C 在结构上更符合问题，但当前最小实现尚未强于 Exp B
- 因此：
  - Exp B = 强诊断基线
  - Exp C = 下一轮更大测试中最值得写论文的 supervision 方向

## 9.4 GTA same-area supervision 比较（`teacher_query_limit=2000`）

该运行位于：

- `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/summary.md`

Official evaluator summary：

| 变体 | Dis@1 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse baseline | 77.16 | 6.42 | 16.58 | 33.78 | 28.35% | 14.20% | 0.29% |
| Exp A current teacher | 70.27 | 8.31 | 19.20 | 39.44 | 19.40% | 14.00% | 0.32% |
| Exp B clean30 | 69.42 | 7.73 | 19.14 | 38.63 | 22.07% | 13.42% | 0.41% |
| Exp C weighted useful-angle | 63.03 | 8.74 | 21.64 | 42.17 | 17.51% | 11.56% | 0.15% |

Teacher 侧训练摘要：

| 变体 | kept / raw | removed | pair weight mode | pair weight mean |
|---|---:|---:|---|---:|
| Exp A current teacher | 2000 / 2000 | 0 | uniform | 1.0000 |
| Exp B clean30 | 1701 / 2000 | 299 | uniform | 1.0000 |
| Exp C weighted useful-angle | 2000 / 2000 | 0 | best_distance_sigmoid | 0.7572 |

解释：

- GTA same-area 同样显示出明显的 teacher noise。
- Exp B 有帮助，说明去噪重要。
- 但 Exp C 是当前最好的 GTA supervision 线，因为它同时提升了：
  - 原始 `Dis@1`
  - 更重要的 robustness metrics
- 因此当前 GTA 默认应为：
  - Exp C weighted useful-angle set supervision
- 如果以后只把一条 GTA supervision 配方迁移到更大 teacher subset 或 cross-area，就带：
  - Exp C

## 9.5 GTA same-area Exp C follow-up（`teacher_query_limit=2000`）

该运行位于：

- `Game4Loc/work_dir/gta_exp_c_followup_runs/exp_c_followup_samearea_q2000_20260410/summary.md`

Official evaluator summary：

| 变体 | Dis@1 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp C baseline (`delta=5`, `center=30`) | 63.03 | 8.74 | 21.64 | 42.17 | 17.51% | 11.56% | 0.15% |
| Exp C1 (`useful_delta_m=3`) | 63.82 | 8.74 | 20.94 | 42.23 | 16.67% | 12.69% | 0.17% |
| Exp C2 (`pair_weight_center_m=20`) | 62.72 | 8.31 | 21.49 | 41.94 | 19.05% | 10.75% | 0.23% |

Teacher 侧摘要：

| 变体 | useful-set mean | pair-weight mean | note |
|---|---:|---:|---|
| Exp C baseline (`delta=5`, `center=30`) | 3.7090 | 0.7572 | current default |
| Exp C1 (`useful_delta_m=3`) | 2.8675 | 0.7572 | useful set became tighter |
| Exp C2 (`pair_weight_center_m=20`) | 3.7090 | 0.6141 | weighting became stricter |

解释：

- Exp C 对 useful-angle set 定义的敏感性似乎高于对 pair weighting 的敏感性。
- 对于 GTA same-area，`useful_delta_m=3` **并不**比 `5` 更好。
- `pair_weight_center_m=20` **也不是** `30` 的更稳定替代。
- 因此保持当前 GTA 默认：
  - `useful_delta_m = 5`
  - `pair_weight_center_m = 30`
- 如果以后继续做 GTA，应先扩大 teacher subset，再继续微调这两个 knob。

## 9.6 GTA-UAV same-area 面向论文主表状态

当前 GTA same-area 面向论文的行集为：

1. dense DKM
2. sparse
3. sparse + rotate90 + inlier-count selection
4. sparse + VOP

当前已完成日志：

- dense DKM:
  - `Game4Loc/work_dir/gta_samearea_dense_shards_20260412/merged_summary.md`
- sparse:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2000.log`
- sparse + rotate90 + inlier-count selection:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2008.log`
- sparse + VOP:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260411_2028.log`

Official evaluator summary：

| 变体 | Recall@1 | Recall@5 | Recall@10 | mAP | Dis@1 | Dis@3 | Dis@5 | MA@3 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds | mean inliers | mean inlier ratio | mean total time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense DKM | 91.11 | 99.39 | 99.54 | 94.81 | 50.11 | 165.20 | 216.94 | 5.72 | 13.51 | 29.74 | 54.81 | 1.57% | 12.00% | 1.51% | 4225.88 | 0.8757 | 4.0410s |
| sparse | 91.11 | 99.39 | 99.54 | 94.81 | 108.47 | 165.20 | 216.94 | 0.61 | 1.86 | 5.61 | 14.61 | 58.41% | 18.47% | 0.29% | 19.56 | 0.1374 | 0.0651s |
| sparse + rotate90 + inlier-count | 91.11 | 99.39 | 99.54 | 94.81 | 77.50 | 165.20 | 216.94 | 3.22 | 8.07 | 19.02 | 36.45 | 11.97% | 21.49% | 0.58% | 53.28 | 0.2036 | 0.2831s |
| sparse + VOP | 91.11 | 99.39 | 99.54 | 94.81 | 62.59 | 165.20 | 216.94 | 3.78 | 8.34 | 22.19 | 43.54 | 12.02% | 13.30% | 0.26% | 70.06 | 0.2912 | 0.3044s |

解释：

- dense DKM 现在是完整 GTA same-area 上在以下两方面都最强的一行：
  - 绝对精度
  - 稳健性
- sparse-only 很快，但 fine-localization 精度和稳健性掉得过多。
- rotate90 + inlier-count 是一个强基线，必须在论文中明确保留。
- 在 denser-SP 刷新后，plain sparse **并没有**整体变强；它主要是用更低的 fallback 换来了更差的 `Dis@1` 与更糟的 `worse-than-coarse`。
- sparse + VOP 在当前 full GTA same-area test 上，仍优于刷新后的 rotate baseline，体现在：
  - 原始 `Dis@1`
  - 所有 `MA@{3,5,10,20}`
  - worse-than-coarse
  - inlier 统计
- 相比 dense DKM，sparse + VOP：
  - 原始 `Dis@1` 差约 `12.48m`
  - `MA@20` 差约 `11.27pp`
  - fallback 差约 `10.45pp`
  - 但速度快约 `13.3x`
- 从 runtime 看：
  - VOP 相比刷新后的 rotate baseline 只增加约 `0.021s/query`
  - 但带来了有意义的质量提升

实际结论：

- 当前 GTA 主表证据已经支持：
  - VOP > sparse
  - VOP > 显式 rotate90 baseline
- 同时也支持 matched 的 dense-vs-ours 对比：
  - dense DKM 在绝对质量上仍更强
  - sparse + VOP 仍快得多，并且仍是最强 sparse 线

## 9.7 UAV-VisLoc `same-area-paper7` 面向论文主表

当前面向论文的运行目录：

- `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411`

训练产物：

- teacher cache:
  - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/teacher_samearea_paper7.pt`
- VOP checkpoint:
  - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/vop_samearea_paper7_useful5_weight30_e6.pth`

Official evaluator logs：

- dense:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_0400.log`
- sparse:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2053.log`
- sparse + rotate90 + inlier-count selection:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2058.log`
- sparse + VOP:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2103.log`

四行的 retrieval metrics 固定一致：

- `Recall@1 = 65.27`
- `Recall@5 = 87.99`
- `Recall@10 = 91.64`
- `AP = 75.29`

Official evaluator summary：

| 变体 | Dis@1 | Dis@3 | Dis@5 | MA@3 | MA@5 | MA@10 | MA@20 | fallback | worse-than-coarse | out-of-bounds | mean inliers | mean inlier ratio | mean total time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense DKM | 241.40 | 399.49 | 483.15 | 2.35 | 4.44 | 15.14 | 35.51 | 8.09% | 48.04% | 7.83% | 3150.2 | 0.6533 | 4.1128s |
| sparse | 274.68 | 399.49 | 483.15 | 0.00 | 1.31 | 4.96 | 10.44 | 43.60% | 78.59% | 1.04% | 19.2 | 0.1332 | 0.0747s |
| sparse + rotate90 + inlier-count | 258.18 | 399.49 | 483.15 | 0.78 | 1.83 | 5.22 | 18.28 | 16.97% | 62.40% | 0.78% | 34.9 | 0.1701 | 0.3111s |
| sparse + VOP | 257.94 | 399.49 | 483.15 | 1.57 | 3.13 | 8.62 | 25.59 | 15.93% | 60.57% | 0.78% | 38.2 | 0.1851 | 0.3291s |

解释：

- sparse-only 比 dense 快很多，但稳健性损失过大。
- rotate90 + inlier-count 是必要基线，因为它恢复了 sparse-only 下跌中的相当一部分。
- 在 denser-SP 刷新后，plain sparse 的 headline accuracy 变差了，因此这个新默认**不能**被包装成通用 sparse 改进。
- sparse + VOP 在 Paper7 上依然优于刷新后的 rotate baseline，体现在：
  - 原始 `Dis@1`
  - `MA@3`
  - `MA@5`
  - `MA@10`
  - `MA@20`
  - fallback
  - worse-than-coarse
- 从 runtime 看，在 Paper7 上 sparse + VOP 和 rotate baseline 几乎同成本，仅多约 `0.018s/query`。

实际结论：

- 在当前 Paper7 上：
  - VOP 明显优于 sparse baselines
  - 但在 headline accuracy / robustness 上仍落后于 dense DKM
- 因此当前 Paper7 证据支持如下论文主张：
  - VOP 是被比较的 sparse baselines 中最强的 sparse fine-localization 变体
- 但它**不支持**更强的说法：
  - VOP 在 Paper7 上已经追平或超过 dense DKM

## 9.8 补充性外部 matcher baseline：LoFTR

LoFTR 现在可通过以下方式作为补充 official-evaluator baseline 使用：

- `eval_visloc.py --loftr`
- `eval_gta.py --loftr`

当前已完成的摘要文件：

- UAV-VisLoc Paper7:
  - `Game4Loc/work_dir/loftr_baseline_runs/visloc_paper7_loftr_20260411/summary.md`
- GTA-UAV same-area:
  - `Game4Loc/work_dir/loftr_baseline_runs/gta_samearea_loftr_20260411/summary.md`

Official evaluator summary：

| 数据集 | 变体 | Dis@1 | MA@20 | fallback | worse-than-coarse | mean total time |
|---|---|---:|---:|---:|---:|---:|
| UAV-VisLoc Paper7 | LoFTR | 277.03 | 19.06 | 7.31% | 65.54% | 0.6796s |
| GTA same-area | LoFTR | 130.66 | 16.93 | 4.01% | 54.98% | 0.6972s |

解释：

- LoFTR 相比 raw sparse baseline 往往能降低 fallback。
- 但在当前正式协议上，它**并没有**超过：
  - dense DKM
  - sparse + rotate90 + inlier-count
  - sparse + VOP
- 因此 LoFTR 当前应被视为：
  - 补充性的外部 baseline
  - **不是**当前 sparse 论文主线的替代方案

重要实现说明：

- 当前稳定的 LoFTR 路径使用：
  - Kornia pretrained outdoor LoFTR
  - matcher 路径内部的 homography-only RANSAC

## 9.9 Sparse matcher-control 的结论（VisLoc yaw 对齐，无 VOP）

这些运行只用于受控的 matcher 分析，不是论文主表协议。

当前摘要文件：

- `Game4Loc/work_dir/visloc_sparse_yaw_matcher_control_runs/visloc_sparse_yaw_matcher_control_20260410_v2/summary.md`
- `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/visloc_sparse_yaw_scale_contrib_20260411_005234/summary.md`
- `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/dense_down_dedup3_followup_20260411_011421/summary.md`
- quick `dedup2 / dedup5` follow-up logs:
  - `Game4Loc/work_dir/visloc_sparse_yaw_scale_contrib_runs/dense_down_dedup_radius_quick_20260411_012113/`

受控 quarter-subset 亮点：

| 变体 | Dis@1 | MA@20 | fallback | mean inliers | mean total time |
|---|---:|---:|---:|---:|---:|
| baseline | 69.38 | 6.90 | 62.07% | 16.3 | 0.1158s |
| LightGlue official-style profile | 73.88 | 3.45 | 100.00% | 0.0 | 0.0888s |
| query-only multi-scale | 92.73 | 6.90 | 20.69% | 22.0 | 0.0683s |
| gallery-only multi-scale | 71.50 | 10.34 | 62.07% | 14.8 | 0.0708s |
| baseline + dedup5 | 69.73 | 6.90 | 68.97% | 13.7 | 0.0735s |

更密 pyramid 的亮点：

| 变体 | Dis@1 | MA@20 | fallback | mean inliers | mean total time |
|---|---:|---:|---:|---:|---:|
| baseline | 70.87 | 6.90 | 68.97% | 15.6 | 0.0885s |
| dense_down | 85.57 | 10.34 | 6.90% | 27.7 | 0.1335s |
| dense_mix_up | 97.42 | 6.90 | 10.34% | 33.0 | 0.1285s |
| dense_down + dedup3 | 91.59 | 3.45 | 17.24% | 20.4 | 0.1155s |
| dense_down + dedup2 | 82.94 | 3.45 | 34.48% | 18.1 | 0.1189s |
| dense_down + dedup5 | 79.32 | 10.34 | 27.59% | 19.8 | 0.1192s |

解释：

- 更多 retained matches / 更多 inliers **不会**自动改善最终定位距离。
- 更密的 pyramid 常常减少 fallback、提升 inlier 数，但依然会恶化 `Dis@1`。
- cross-scale dedup 能在一定程度上修复 dense-down 变体，其中快速测试过的半径里 `5px` 看起来最好，但仍没超过 baseline。
- 因此这个较早的 VisLoc yaw 对齐控制研究目前只支持：
  - 保持当前 LightGlue profile
  - 保持当前 multi-scale policy
  - 默认不启用 cross-scale dedup
- 后来 denser-SuperPoint 默认升级的依据，来自 GTA / Paper7 在当前论文主线上的刷新结果，而不是这些更早的 yaw 对齐、无 VOP 控制实验。

## 9.10 来自外部 SP+LG 项目的 GTA sparse matcher-control 迁移检查

这些运行只是在当前 `sparse + VOP` 线上做的 **小规模 GTA same-area 控制实验**，不是正式论文主表行。

目的：

- 测试另一个项目中的更强 SuperPoint+LightGlue 配方，是否能迁移到当前 GTA `VOP + sparse` 流水线
- 特别是：
  - 更密的 SuperPoint detection
  - 不同的 LightGlue thresholding
  - 更紧的 geometry threshold
  - cross-scale dedup

匹配设置：

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
  - 之所以选它，是因为它比旧的 `first-172` 前缀更能跟踪 GTA full-run accuracy

相关日志：

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

控制摘要：

| 变体 | 相对 baseline 的精确改动 | Dis@1 | MA@20 | fallback | worse-than-coarse | mean inliers | mean total time |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | current GTA sparse + VOP | 62.90 | 39.42 | 21.16% | 10.14% | 51.54 | 0.2894s |
| external-style LG transfer | `SP: det=3e-4, kp=4096` + `LG=minima_ref` + `dedup5` + `H thr=8` | 104.80 | 6.09 | 100.00% | 0.00% | 0.00 | 0.3345s |
| denser SP + dedup5 + tight H | `SP: det=3e-4, kp=4096` + `dedup5` + `H thr=8` | 61.21 | 37.68 | 34.49% | 4.06% | 42.90 | 0.3021s |
| denser SP + dedup5 | `SP: det=3e-4, kp=4096` + `dedup5` | 60.39 | 41.74 | 12.46% | 12.17% | 63.54 | 0.2957s |
| denser SP only | `SP: det=3e-4, kp=4096` | 59.47 | 43.19 | 12.75% | 13.91% | 78.88 | 0.2941s |

解释：

- 外部项目的 LightGlue 风格 profile **并没有**干净迁移过来。
  它几乎完全退化为 fallback，不能采用。
- 主要失败模式**不是**“更多 keypoints 有害”。
  真正有害的迁移是：
  - LightGlue profile 的改变本身
  - 更紧的 geometry threshold（`H thr=8`）
- 在当前 GTA `VOP + sparse` 上，更稠密的 SuperPoint 其实是有希望的：
  - `Dis@1` 从 `62.90m` 改善到 `59.47m`
  - `MA@20` 从 `39.42%` 提升到 `43.19%`
  - fallback 从 `21.16%` 降到 `12.75%`
  - runtime 几乎不变
- `5px` 的 cross-scale dedup **不是**这里的关键增益来源。
  在这个 GTA 子集上，`denser SP only` 还略优于 `denser SP + dedup5`。
- 权衡在于：更密 SP 也略微增加了 `worse-than-coarse`，所以仅凭这组小规模控制结果还不足以支撑论文结论。

当前实际结论：

- 这条 matcher-level follow-up 现已在正式 GTA / Paper7 协议上提升并刷新。
- 那些更大规模运行的结果是：
  - 对当前 `sparse + VOP` 主线可接受到略有正效应
  - 但不是 plain sparse 的通用胜利
- 当前**不要**推广：
  - 外部 LightGlue profile 迁移
  - 从外部项目复制来的更紧 homography threshold
  - 将 cross-scale dedup 作为新默认值

## 9.11 正式协议上的 denser SuperPoint 默认刷新

由用户指定的默认刷新：

- sparse matcher 默认现在使用：
  - `SuperPoint detection_threshold = 0.0003`
  - `SuperPoint max_num_keypoints = 4096`

刷新后的官方运行：

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

跨数据集解读：

- 在两个正式协议上，论文主线仍成立：
  - `sparse + VOP` 仍然是 sparse 行中最强的 sparse fine-localization 变体
- 刷新后的默认值并**没有**让 plain sparse 成为更强的论文行。
- 这一刷新后的默认值与以下路线最兼容：
  - rotate-aware sparse
  - `VOP + sparse`
- 因此未来的 matched experiments 应把当前 sparse 默认视为：
  - denser SuperPoint
  - LightGlue profile 不变
  - cross-scale dedup 不变
  - multi-scale policy 不变

## 9.12 03/04 开发 follow-up：`VOP + sparse` 的带门控双路径 geometry retry

这些运行仍然 **仅用于开发**，并使用旧的 `03/04 same-area` 协议。不要把它们作为论文 headline result 引用。

目的：

- 在当前 `VOP + sparse` 线上继续做 post-retrieval fine-localization diagnosis
- 测试能否在使用 `USAC_MAGSAC` 作为主 geometry estimator 的同时，只在 fallback 情况下保守地使用 `RANSAC` retry

固定设置：

- UAV-VisLoc `03/04 same-area`
- retrieval 固定为：
  - `Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`
- VOP 固定为：
  - `Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- mode:
  - `with_match + sparse + prior_topk=4`
- no yaw

当前关键日志：

- 当前 denser-SP 默认下的 baseline:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260411_2358.log`
- primary `USAC_MAGSAC`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0018.log`
- pure `USAC_MAGSAC` matched rerun:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0808.log`
- 双路径 `per_candidate`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0745.log`
- 双路径 `final_only`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0754.log`
- 带 `accept 25 / 0.15` 的双路径 `final_only`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0801.log`
- 带 `accept 20 / 0.10` 的双路径 `final_only`:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_VisLoc_same_match_on_VisLoc_20260412_0805.log`

摘要表：

| 变体 | Dis@1 | MA@20 | fallback | worse-than-coarse | secondary_takeover | mean total time |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 50.25 | 30.17 | 7.76% | 40.52% | 0.00% | 0.3343s |
| `USAC_MAGSAC` | 47.99 | 31.03 | 27.59% | 48.28% | 0.00% | 0.2694s |
| `USAC_MAGSAC` rerun | 49.34 | 31.03 | 29.31% | 49.14% | 0.00% | 0.2651s |
| two-path `per_candidate` | 53.35 | 26.72 | 0.86% | 40.52% | 35.34% | 0.4331s |
| two-path `final_only` | 51.50 | 31.03 | 10.34% | 42.24% | 16.38% | 0.2888s |
| two-path `final_only` + `accept 25 / 0.15` | 50.62 | 29.31 | 36.21% | 52.59% | 0.00% | 0.2787s |
| two-path `final_only` + `accept 20 / 0.10` | 46.49 | 35.34 | 19.83% | 45.69% | 2.59% | 0.2822s |

解释：

- `USAC_MAGSAC` 仍是唯一一个相对当前 baseline 能提升原始 `Dis@1` 的单因素几何改动。
- 但朴素的 fallback 修复并不够：
  - `per_candidate` retry 过度修正，严重破坏 ranking
  - 不加门控的 `final_only` retry 更干净，但仍不够强
- 第一个明显有用的保守变体是：
  - primary:
    - `USAC_MAGSAC`
  - secondary:
    - `RANSAC`
  - retry mode:
    - `final_only`
  - secondary acceptance:
    - `min_inliers >= 20`
    - `min_inlier_ratio >= 0.10`
- 在当前 matched rerun 中，这个变体同时优于：
  - 当前 denser-SP baseline
  - pure `USAC_MAGSAC` rerun
- 因此，如果以后还要继续做 03/04 开发，这条最有前景的下一分支应是：
  - 保持正式论文默认值不变
  - 在本地开发中继续推进带门控的 `final_only` 双路径 geometry retry

## 9.13 GTA same-area 上对带门控双路径 geometry retry 的快速迁移检查

这些运行只是在 GTA-UAV 上做的 **小规模 matched same-area experiment**。它们不是正式主表行。

目的：

- 测试当前在 `03/04` 上最好的开发方向，能否在不改 retrieval、也不改 matcher internals 的前提下迁移到 GTA same-area
- 设置固定为当前 GTA sparse 主线：
  - retrieval checkpoint:
    - `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`
  - VOP checkpoint:
    - `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/artifacts/exp_c_useful5_weight30_e6.pth`
  - `with_match + sparse + prior_topk=4`
  - 相同 query subset:
    - `query_limit=345`

当前日志：

- baseline subset:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260412_0820.log`
- pure `USAC_MAGSAC` subset:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260412_0830.log`
- gated dual-path subset:
  - `Game4Loc/Log/vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260412_0826.log`

摘要表：

| 变体 | Dis@1 | MA@20 | fallback | worse-than-coarse | secondary_takeover | mean total time |
|---|---:|---:|---:|---:|---:|---:|
| GTA subset baseline | 58.33 | 45.22 | 10.72% | 13.04% | 0.00% | 0.2536s |
| GTA subset `USAC_MAGSAC` | 58.63 | 41.16 | 20.29% | 7.25% | 0.00% | 0.2400s |
| GTA subset gated dual-path | 57.60 | 41.16 | 22.32% | 7.25% | 0.87% | 0.2396s |

带门控双路径变体细节：

- primary:
  - `USAC_MAGSAC`
- secondary:
  - `RANSAC`
- mode:
  - `final_only`
- acceptance gate:
  - `min_inliers >= 20`
  - `min_inlier_ratio >= 0.10`

解释：

- `03/04` 上的开发方向**没有**干净迁移到 GTA same-area。
- 在这个 matched subset 上，带门控双路径变体相对 GTA baseline 只带来了很小的原始 `Dis@1` 改善：
  - `58.33m -> 57.60m`
- 但它明显恶化了更重要的稳健性 / coverage 指标：
  - `MA@20`: `45.22% -> 41.16%`
  - `fallback`: `10.72% -> 22.32%`
- 因而 GTA 上的效应与 `03/04` 开发协议不同：
  - `USAC_MAGSAC` 风格几何会降低 `worse-than-coarse`
  - 但也会把更多 query 推入 fallback
- 实际结论：
  - 将这条线保留为 **开发期 GTA 诊断**
  - **不要**把它提升为 GTA sparse 默认值
- 如果以后继续做 GTA geometry 工作，下一个有价值步骤不是继续复制 retry heuristics，而是理解为什么 MAGSAC 在 same-area GTA 上会如此明显地提高 fallback


# 10. 必须报告的关键指标

对于 official fine-localization experiments，始终报告：

- 如果日志中已经提供，则报告 `Recall@1`、`Recall@5`、`Recall@10`、`AP`
- `Dis@1`、`Dis@3`、`Dis@5`
- `MA@3m`、`MA@5m`、`MA@10m`、`MA@20m`
- `worse-than-coarse` 的 count / ratio
- `fallback` 的 count / ratio
- `identity-H fallback` 的 count
- `out-of-bounds` 的 count
- `projection-invalid` 的 count
- mean retained matches
- mean inliers
- mean inlier ratio
- mean VOP forward time / query
- mean matcher time / query
- mean total fine-localization time / query

对于 cached mechanism experiments，报告：

- top-k oracle-best coverage
- useful-angle coverage
- useful-angle set recall
- covered-vs-missed final error


# 11. Checkpoints 与产物

## 11.1 Retrieval / backbone checkpoint

当前做可比实验时使用：

- `Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`

当前面向 UAV-VisLoc `same-area-paper7` 的论文实验使用：

- `Game4Loc/work_dir/visloc/vit_base_patch16_rope_reg1_gap_256.sbb_in1k/0409152642/weights_e10_0.6527.pth`

当前 GTA-UAV 可比实验使用：

- same-area:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`
- cross-area:
  - `Game4Loc/pretrained/gta/vit_base_eva_gta_cross_area.pth`

除非明确必要，否则不要在新的 matched run 中使用：

- `Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_best.pth`
  - 更旧 / 训练不完整的状态

## 11.2 Orientation checkpoints

- 当前冻结 baseline：
  - `Game4Loc/work_dir/vop/vop_0407_full_rankce_e6.pth`
- Hard clean-pair baseline：
  - `Game4Loc/work_dir/vop/vop_0409_clean30_rankce_e6.pth`
- Weighted useful-angle baseline：
  - `Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- 当前 UAV-VisLoc `same-area-paper7` 面向论文的 checkpoint：
  - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/vop_samearea_paper7_useful5_weight30_e6.pth`
- 当前 GTA-UAV same-area 面向论文的 checkpoint：
  - `Game4Loc/work_dir/gta_vop_samearea_supervision_compare_runs/gta_samearea_supervision_compare_q2000_20260410/artifacts/exp_c_useful5_weight30_e6.pth`

## 11.3 Teacher cache

- `Game4Loc/work_dir/vop/teacher_0407_full.pt`
- 当前 UAV-VisLoc `same-area-paper7` teacher cache：
  - `Game4Loc/work_dir/paper7_main_table_runs/visloc_paper7_20260411/artifacts/teacher_samearea_paper7.pt`

## 11.4 Confidence verifier artifact

它确实存在，但 **仅用于 ablation**：

- `Game4Loc/work_dir/confidence/linear_verifier_same_area_prior_topk4.pth`

## 11.5 Cached mechanism files

当前有用的 cached 文件：

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

这些只用于 linkage check：

- `Game4Loc/work_dir/vop/gta_samearea_smoke_teacher.pt`
- `Game4Loc/work_dir/vop/gta_samearea_smoke_vop.pth`

**不要**用它们做任何结论。

## 11.7 补充 matcher-baseline 摘要

当前有用的补充 baseline 摘要：

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


# 12. 当前 fine-localization 主线的文件地图

核心 evaluator 与 matcher 路径：

- `Game4Loc/eval_visloc.py`
- `Game4Loc/game4loc/evaluate/visloc.py`
- `Game4Loc/game4loc/matcher/sparse_sp_lg.py`
- `Game4Loc/game4loc/matcher/gim_dkm.py`
- `Game4Loc/game4loc/dataset/visloc.py`
- `Game4Loc/game4loc/models/model.py`

Orientation / training utilities：

- `Game4Loc/game4loc/orientation/vop.py`
- `Game4Loc/build_vop_teacher.py`
- `Game4Loc/train_vop.py`
- `Game4Loc/build_topk_cache.py`
- `Game4Loc/eval_topk_cached.py`
- `Game4Loc/analyze_topk_hypotheses.py`
- `Game4Loc/analyze_vop.py`
- `Game4Loc/train_confidence_verifier.py`

GTA-UAV evaluation 路径：

- `Game4Loc/eval_gta.py`
- `Game4Loc/game4loc/evaluate/gta.py`
- `Game4Loc/game4loc/dataset/gta.py`


# 13. 可复现实验命令

以下所有命令默认：

- cwd: `/home/lcy/Workplace/GTA-UAV/Game4Loc`
- env: `gtauav`

## 13.1 03/04 开发协议上的 official evaluator baselines

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

### 当前冻结 proposer，`prior_topk=4`

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

### 当前冻结 proposer，`prior_topk=2`

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

## 13.2 训练 supervision diagnosis 变体

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

## 13.3 Supervision diagnosis checkpoints 的 official evaluator

### clean30，`prior_topk=4`

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

### useful-weighted，`prior_topk=4`

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

## 13.5 GTA-UAV VOP 迁移

重要 runtime 说明：

- 优先：
  - `--num_workers 0`
- 优先：
  - `--with_match --sparse`

当前 GTA 默认训练配方：

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

当前一键 GTA same-area pipeline 默认：

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

当前行为说明：

- 该命令会把标准 evaluation log 写到：
  - `Game4Loc/Log/...`
- 默认**不会**写逐 query 的 match 文件

### GTA cross-area 切换

要把同一流水线切换到 cross-area，需要同时切两处：

- checkpoint:
  - `./pretrained/gta/vit_base_eva_gta_cross_area.pth`
- meta files:
  - `cross-area-drone2sate-train.json`
  - `cross-area-drone2sate-test.json`

## 13.6 当前面向论文的主表命令

这些命令是当前面向论文的比较命令。

保持 retrieval 固定。只改变 fine-localization module / matcher path。

### GTA-UAV same-area 主表

使用 retrieval checkpoint：

- `Game4Loc/pretrained/gta/vit_base_eva_gta_same_area.pth`

#### DKM dense matching（原始 matcher 路径）

```bash
WANDB_MODE=disabled /home/lcy/miniconda3/envs/gtauav/bin/python eval_gta.py \
  --data_root ./data/GTA-UAV-data \
  --test_pairs_meta_file same-area-drone2sate-test.json \
  --model vit_base_patch16_rope_reg1_gap_256.sbb_in1k \
  --checkpoint_start ./pretrained/gta/vit_base_eva_gta_same_area.pth \
  --with_match --dense --num_workers 0
```

重要说明：

- 这是原始 dense DKM 路径
- 在这台机器上跑 full same-area GTA 会非常慢

#### Sparse baseline（默认 multi-scale，无 rotate）

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

当前 evaluator 对这一行的行为：

- 默认启用 sparse multi-scale
- 如果省略 `--no_rotate`，默认会进行 rotate search
- candidate selection 规则是：
  - 最优 `inlier count`
  - 并列时按 `inlier ratio`

#### Sparse + VOP（ours）

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

### UAV-VisLoc `same-area-paper7` 主表

使用 retrieval checkpoint：

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

#### Train paper7 VOP（Exp C）

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

#### Sparse baseline（默认 multi-scale，无 rotate）

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

当前 evaluator 对这一行的行为：

- 默认启用 sparse multi-scale
- 保持 `--sparse_angle_score_inlier_offset` 不设置
- candidate selection 规则是：
  - 最优 `inlier count`
  - 并列时按 `inlier ratio`

#### Sparse + VOP（ours）

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

## 13.7 补充性外部 matcher baseline 命令

这些命令用于补充比较，不替代四个核心主表行。

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


# 14. 下一个 agent 在论文中应该如何表述

推荐的论文 framing：

1. **问题诊断**
   - post-retrieval fine localization 之所以脆弱，是因为朝向歧义与不可靠对应关系、不稳定几何发生了耦合

2. **方法原则**
   - 与其强行预测单个 best angle，不如预测一小组有用角度假设

3. **结构化解决方案**
   - 先做 useful-angle proposal
   - 再做 geometry verification

4. **监督改进**
   - noisy teacher signal 很重要
   - 监督应从单角度误差蒸馏，转向带 pair confidence 的 useful-angle set supervision

**不要**把当前工作表述为：

> “我们尝试了很多 heuristics，最后找到一个有效的”


# 15. 下一个 agent 绝对不能宣称什么

**不要**宣称：

- “SOTA on UAV-VisLoc”
- “在与原论文匹配的协议下取胜”
- “full-protocol superiority”

除非新的实验确实复现了匹配参考协议。

当前 03/04 的数字还不足以支持这些说法。


# 16. 当前最推荐的下一步

如果下一个 agent 必须只选一个严谨的下一动作，应当是：

> 将 **pair-confidence-weighted useful-angle set supervision** 这条线迁移到更大、更严格的 UAV-VisLoc strict-pos 协议上，并与以下对象比较：
> - 当前冻结 top-k baseline
> - hard clean-pair 诊断基线

**不要**重新打开以下方向：

- single-angle VOP rescue
- teacher-only soft posterior distillation
- partial unfreezing
- 大规模 heuristic sweep

除非用户明确要求。


# 17. 未来实验摘要必须使用的输出格式

每个完成的实验摘要都必须使用：

## Experiment Name
- 一行说明目的

## Change Compared to Baseline
- 只写精确改动

## Quantitative Results
- 主要指标
- runtime
- 重要 count / statistics

## Interpretation
- 可能是什么提升了
- 可能什么没有提升
- 结果看起来是稳健还是脆弱

## Decision
必须且只能选一个：
- `KEEP`
- `REJECT`
- `NEEDS ONE FOLLOW-UP`
- `GOOD FOR PAPER ABLATION ONLY`

不要用含糊的结尾。


# 18. 最终原则

这个仓库不应变成一个角度扫描和救火逻辑的博物馆。

优先选择：

- 一个更干净、研究故事更强的方法

而不是：

- 一个指标略好、但更杂乱的补丁

除非证据压倒性地说明后者更值得保留。
