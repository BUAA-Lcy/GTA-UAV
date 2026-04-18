# Schematic Paper Figure Modules

This folder contains the no-text, icon-like module library for the VOP paper figures.

- Total exported modules: `34`
- From real local images or real computed results: `26`
- Redrawn / vectorized schematic modules: `8`

## Notes

- All module PNGs use transparent backgrounds.
- Module images are intentionally text-free so they can be re-labeled in PPT / Illustrator later.
- Preview sheets keep light captions only for human inspection.

## Missing Inputs

- No required input file is missing.

## Module Inventory

### `uav_input_tile_icon.png`
- category: `input`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG`
- real-derived: `True`
- usage: Use for the UAV query image in a paper diagram.
- description: Real UAV input tile, simplified as a text-free icon module.

### `satellite_input_tile_icon.png`
- category: `input`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png`
- real-derived: `True`
- usage: Use for the gallery / retrieved satellite input block.
- description: Real satellite top-1 tile, simplified as a text-free icon module.

### `input_pair_icon.png`
- category: `input`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png`
- real-derived: `True`
- usage: Use when a compact pair module is easier than placing q and g separately.
- description: Simplified side-by-side input pair icon built from the real query-gallery pair.

### `final_localization_basemap_icon.png`
- category: `input`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/01_5_005_022.png`
- real-derived: `True`
- usage: Use as the result-stage satellite basemap block.
- description: Raw satellite basemap tile used under the final localization result.

### `frozen_backbone_icon.png`
- category: `backbone`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`
- real-derived: `True`
- usage: Use as the backbone stage icon without any embedded labels.
- description: Minimal frozen-backbone schematic driven by the real query, gallery, and feature maps.

### `learnable_vop_module_icon.png`
- category: `backbone`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- real-derived: `True`
- usage: Use as the VOP stage outer box in a diagram.
- description: Minimal VOP schematic built from real encoded maps and the real posterior.

### `feature_map_f_q_icon.png`
- category: `features`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`
- real-derived: `True`
- usage: Use as the F_q feature block.
- description: Raw query feature map visualized with PCA colors.

### `feature_map_f_g_icon.png`
- category: `features`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth`
- real-derived: `True`
- usage: Use as the F_g feature block.
- description: Raw gallery feature map visualized with PCA colors.

### `feature_map_f_q_rot_icon.png`
- category: `features`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- real-derived: `True`
- usage: Use as the F_{q,rot} feature block.
- description: Rotated query feature map from the real selected VOP angle.

### `fusion_block_f_g_icon.png`
- category: `features`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- real-derived: `True`
- usage: Use as the [F_g] fusion input.
- description: Encoded gallery feature tile for the VOP fusion stage.

### `fusion_block_f_q_rot_icon.png`
- category: `features`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- real-derived: `True`
- usage: Use as the [F_{q,rot}] fusion input.
- description: Encoded rotated-query feature tile for the VOP fusion stage.

### `fusion_block_f_g_mul_f_q_rot_icon.png`
- category: `features`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- real-derived: `True`
- usage: Use as the [F_g ⊙ F_{q,rot}] block.
- description: Element-wise product fusion tile from the real VOP sample.

### `fusion_block_abs_f_g_minus_f_q_rot_icon.png`
- category: `features`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- real-derived: `True`
- usage: Use as the [|F_g - F_{q,rot}|] block.
- description: Absolute-difference fusion tile from the real VOP sample.

### `feature_concat_icon.png`
- category: `features`
- source: `vector redraw`
- real-derived: `False`
- usage: Place between fusion inputs and the VOP head.
- description: No-text schematic icon for feature concatenation / fusion.

### `conv_relu_gap_head_icon.png`
- category: `features`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth`
- real-derived: `True`
- usage: Use as the learnable VOP scoring head.
- description: Minimal head icon showing the fusion preview feeding a three-block head.

### `rotation_candidates_icon.png`
- category: `features`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG`
- real-derived: `True`
- usage: Use for the candidate-angle generation step.
- description: Real query tile wrapped by a rotation orbit schematic.

### `posterior_probability_distribution_icon.png`
- category: `posterior`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/topk_analysis/posterior_k4.json`
- real-derived: `True`
- usage: Use for the posterior output block.
- description: Real VOP posterior rendered as a no-text polar icon.

### `topk_selected_angles_icon.png`
- category: `posterior`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG`
- real-derived: `True`
- usage: Use for the selected top-k hypotheses step.
- description: Real top-k selected angle hypotheses shown as rotated query thumbnails.

### `soft_orientation_target_icon.png`
- category: `posterior`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-test.json`
- real-derived: `True`
- usage: Use in the supervision branch of a figure.
- description: Real teacher soft target re-rendered without titles or axis text.

### `useful_angle_set_icon.png`
- category: `posterior`
- source: `03_0570.JPG`
- real-derived: `True`
- usage: Use when illustrating multimodal useful-angle structure.
- description: Real multimodal useful-angle example re-rendered as a no-text plot icon.

### `pair_confidence_weight_icon.png`
- category: `posterior`
- source: `real training recipe: center=30, scale=10`
- real-derived: `False`
- usage: Use in training-supervision schematics.
- description: Minimal pair-weight sigmoid icon from the current training recipe.

### `superpoint_lightglue_geometric_verification_icon.png`
- category: `verification`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png`
- real-derived: `True`
- usage: Use as the downstream geometric verification stage.
- description: Verification-stage panel containing the real VOP-guided sparse match visual.

### `sparse_no_vop_match_icon.png`
- category: `verification`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG`
- real-derived: `True`
- usage: Use for the plain sparse branch.
- description: Real sparse matching visual without VOP prior.

### `ours_vop_guided_match_icon.png`
- category: `verification`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG`
- real-derived: `True`
- usage: Use for the Ours branch.
- description: Real VOP-guided sparse matching visual.

### `match_visual_comparison_icon.png`
- category: `verification`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG`
- real-derived: `True`
- usage: Use when one block should summarize the match-quality difference.
- description: Compact side-by-side comparison between sparse and VOP-guided matches.

### `final_localization_result_icon.png`
- category: `results`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/figures/pair_vop_assets_20260414/pair01_vop_summary.json`
- real-derived: `True`
- usage: Use as the final result block.
- description: Final localization result with the real projected pin on the satellite tile.

### `accuracy_efficiency_tradeoff_icon.png`
- category: `results`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/figures/vop_shortpaper_20260411/fig02b_dis1_runtime_tradeoff_sp_lg_frontier.png`
- real-derived: `True`
- usage: Use as a compact quantitative inset.
- description: Minimal no-text redraw of the current runtime-vs-Dis@1 plot.

### `localization_error_curve_icon.png`
- category: `results`
- source: `/home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/same-area-paper7-drone2sate-test.json`
- real-derived: `True`
- usage: Use as the mechanism-analysis curve block.
- description: No-text redraw of the real teacher localization error curve.

### `uav_topview_icon.png`
- category: `icons`
- source: `vector redraw`
- real-derived: `False`
- usage: Use as a small pictogram in the input stage.
- description: Standalone UAV icon without text.

### `red_pin_icon.png`
- category: `icons`
- source: `vector redraw`
- real-derived: `False`
- usage: Use as a reusable localization marker.
- description: Standalone red location pin icon.

### `arrow_right_icon.png`
- category: `icons`
- source: `vector redraw`
- real-derived: `False`
- usage: Use between modules when hand-assembling a figure.
- description: Standalone right arrow icon.

### `rotation_arrow_icon.png`
- category: `icons`
- source: `vector redraw`
- real-derived: `False`
- usage: Use near query rotation modules.
- description: Standalone rotation arrow icon without any label text.

### `softmax_icon.png`
- category: `icons`
- source: `vector redraw`
- real-derived: `False`
- usage: Use near posterior-generation stages.
- description: Standalone softmax-style icon without text.

### `topk_chip_icon.png`
- category: `icons`
- source: `vector redraw`
- real-derived: `False`
- usage: Use near selected-angle stages.
- description: Standalone top-k selection icon without text.
