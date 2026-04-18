# PNG Module Library

- Total exported modules: 34
- Modules derived from local real imagery / real experiment figures: 25
- Vectorized / redrawn modules: 9

## Exported Modules

- `uav_input_tile.png`
  meaning: Real UAV query tile used in the main VOP pipeline example.
  category: input
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG
  derived_from_real_image: true
  recommended_usage: Drag into the leftmost input position in the paper figure.
- `satellite_input_tile.png`
  meaning: Real retrieved satellite tile used in the main VOP pipeline example.
  category: input
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
  derived_from_real_image: true
  recommended_usage: Drag into the gallery/top-1 input position.
- `input_pair_q_g.png`
  meaning: Combined real input pair module with q/g labels.
  category: input
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
  derived_from_real_image: true
  recommended_usage: Use as a compact q-g pair block in PPT/Illustrator.
- `final_localization_basemap.png`
  meaning: Real satellite basemap with the final localization pin position.
  category: input
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/01_5_005_022.png
  derived_from_real_image: true
  recommended_usage: Use as the base map before adding extra result annotations.
- `frozen_vit_base_backbone_box.png`
  meaning: Frozen ViT-Base backbone box with real input thumbnails and real feature maps.
  category: backbone
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth
  derived_from_real_image: true
  recommended_usage: Use as the backbone stage in the method figure.
- `learnable_vop_module_box.png`
  meaning: Learnable VOP module outer box with real pairwise feature maps and real posterior.
  category: backbone
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/topk_analysis/cache_same_area_full.json
  derived_from_real_image: true
  recommended_usage: Use as the central VOP block in the method figure.
- `superpoint_lightglue_geometric_verification_box.png`
  meaning: Sparse verification box with real SP+LG match visual.
  category: verification
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
  derived_from_real_image: true
  recommended_usage: Use as the downstream verification stage.
- `feature_map_f_q.png`
  meaning: Real backbone feature map F_q visualized by PCA projection.
  category: features
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth
  derived_from_real_image: true
  recommended_usage: Use as an individual feature tile or within the backbone box.
- `feature_map_f_g.png`
  meaning: Real backbone feature map F_g visualized by PCA projection.
  category: features
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth
  derived_from_real_image: true
  recommended_usage: Use as an individual feature tile or within the backbone box.
- `feature_map_f_q_rot.png`
  meaning: Real rotated query feature map F_{q,rot} after VOP rotation.
  category: features
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
  derived_from_real_image: true
  recommended_usage: Use before the pairwise fusion stage.
- `fusion_block_f_g.png`
  meaning: Real encoded feature tile [F_g] used inside the VOP pairwise head.
  category: features
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
  derived_from_real_image: true
  recommended_usage: Use as one of the four pairwise fusion inputs.
- `fusion_block_f_q_rot.png`
  meaning: Real encoded feature tile [F_{q,rot}] used inside the VOP pairwise head.
  category: features
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
  derived_from_real_image: true
  recommended_usage: Use as one of the four pairwise fusion inputs.
- `fusion_block_f_g_mul_f_q_rot.png`
  meaning: Real multiplicative fusion tile [F_g ⊙ F_{q,rot}].
  category: features
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
  derived_from_real_image: true
  recommended_usage: Use as one of the four pairwise fusion inputs.
- `fusion_block_abs_f_g_minus_f_q_rot.png`
  meaning: Real absolute-difference fusion tile [|F_g - F_{q,rot}|].
  category: features
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
  derived_from_real_image: true
  recommended_usage: Use as one of the four pairwise fusion inputs.
- `feature_concatenation_fusion_symbol.png`
  meaning: Reusable concat / fusion symbol module for the four VOP feature inputs.
  category: features
  source: redrawn vector icon matched to the paper palette
  derived_from_real_image: false
  recommended_usage: Place between the four feature tiles and the prediction head.
- `conv_relu_gap_head_box.png`
  meaning: Prediction head module showing 1×1 Conv → ReLU → GAP.
  category: features
  source: real fusion preview from /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
  derived_from_real_image: true
  recommended_usage: Use as the VOP prediction head.
- `candidate_angles_rotation_module.png`
  meaning: Rotation operation block based on the real UAV query tile.
  category: features
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG
  derived_from_real_image: true
  recommended_usage: Use to explain 36 candidate angles and feature rotation.
- `posterior_probability_distribution.png`
  meaning: Real posterior distribution on the main sample with highlighted top-4 angles.
  category: posterior
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/topk_analysis/cache_same_area_full.json | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/topk_analysis/posterior_k4.json
  derived_from_real_image: true
  recommended_usage: Use as the posterior module in the pipeline figure.
- `topk_selected_angles.png`
  meaning: Real top-k selected angle badges from the main sample.
  category: posterior
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/topk_analysis/posterior_k4.json
  derived_from_real_image: false
  recommended_usage: Use after the posterior module to denote selected hypotheses.
- `soft_orientation_target.png`
  meaning: Soft orientation target from the real teacher example.
  category: posterior
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/teacher_signal_example_20260414/teacher_pair_soft_orientation_target_v2_smooth.png
  derived_from_real_image: true
  recommended_usage: Use in training supervision figures.
- `useful_angle_set_module.png`
  meaning: Useful-angle set module cropped from the real mechanism figure.
  category: posterior
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/vop_shortpaper_20260415_review/fig04a_angle_surface_mechanism.png
  derived_from_real_image: true
  recommended_usage: Use to explain multimodal useful-angle structure.
- `pair_confidence_weight_curve.png`
  meaning: Pair-confidence weighting curve derived from the real training recipe.
  category: posterior
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/train_vop.py
  derived_from_real_image: false
  recommended_usage: Use alongside useful-angle supervision in the training section.
- `sparse_no_vop_match_visual.png`
  meaning: Real SP+LG match visualization without VOP guidance on the main sample.
  category: verification
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
  derived_from_real_image: true
  recommended_usage: Use as the plain sparse baseline visual.
- `ours_vop_guided_match_visual.png`
  meaning: Real VOP-guided SP+LG match visualization on the main sample.
  category: verification
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | best angle +20
  derived_from_real_image: true
  recommended_usage: Use as the VOP-guided verification visual.
- `match_visual_comparison_dual.png`
  meaning: Side-by-side comparison between no-VOP sparse matching and VOP-guided matching.
  category: verification
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
  derived_from_real_image: true
  recommended_usage: Use when a compact comparison panel is needed.
- `final_localization_result.png`
  meaning: Real final localization result with the actual projected pin on the satellite tile.
  category: results
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/pair_vop_assets_20260414/pair01_vop_summary.json
  derived_from_real_image: true
  recommended_usage: Use as the final result panel.
- `accuracy_efficiency_tradeoff_plot.png`
  meaning: Accuracy-efficiency tradeoff plot from the current paper figure set.
  category: results
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/vop_shortpaper_20260411/fig02b_dis1_runtime_tradeoff_sp_lg_frontier.png
  derived_from_real_image: true
  recommended_usage: Use as the standalone quantitative comparison plot.
- `localization_error_curve.png`
  meaning: Real teacher localization error curve on the Paper7 pair.
  category: results
  source: /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/teacher_signal_example_20260414/teacher_pair_localization_error_curve.png
  derived_from_real_image: true
  recommended_usage: Use as a supervision / motivation plot.
- `uav_topview_icon.png`
  meaning: Reusable vectorized UAV top-view icon.
  category: icons
  source: redrawn vector icon because no local UAV icon asset exists
  derived_from_real_image: false
  recommended_usage: Use as a small visual cue in diagrams or legends.
- `red_pin_icon.png`
  meaning: Reusable red localization pin icon.
  category: icons
  source: redrawn vector icon matched to the paper palette
  derived_from_real_image: false
  recommended_usage: Use on top of localization maps or result panels.
- `arrow_right_icon.png`
  meaning: Reusable straight arrow icon.
  category: icons
  source: redrawn vector icon matched to the paper palette
  derived_from_real_image: false
  recommended_usage: Use between adjacent modules in PowerPoint or Illustrator.
- `rotation_arrow_icon.png`
  meaning: Reusable rotation / 36-angle icon.
  category: icons
  source: redrawn vector icon matched to the paper palette
  derived_from_real_image: false
  recommended_usage: Use near rotation or candidate-angle annotations.
- `topk_badge.png`
  meaning: Reusable top-k badge.
  category: icons
  source: redrawn vector badge
  derived_from_real_image: false
  recommended_usage: Use to annotate selected hypotheses.
- `softmax_badge.png`
  meaning: Reusable softmax badge.
  category: icons
  source: redrawn vector badge
  derived_from_real_image: false
  recommended_usage: Use between the head and the posterior distribution.

## Real-source Modules

- `uav_input_tile.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG
- `satellite_input_tile.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
- `input_pair_q_g.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
- `final_localization_basemap.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/01_5_005_022.png
- `frozen_vit_base_backbone_box.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth
- `learnable_vop_module_box.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/topk_analysis/cache_same_area_full.json
- `superpoint_lightglue_geometric_verification_box.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
- `feature_map_f_q.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth
- `feature_map_f_g.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/pretrained/visloc/vit_base_eva_visloc_same_area_0407.pth
- `feature_map_f_q_rot.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
- `fusion_block_f_g.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
- `fusion_block_f_q_rot.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
- `fusion_block_f_g_mul_f_q_rot.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
- `fusion_block_abs_f_g_minus_f_q_rot.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
- `conv_relu_gap_head_box.png` <- real fusion preview from /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/vop_0409_useful5_weight30_e6.pth
- `candidate_angles_rotation_module.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG
- `posterior_probability_distribution.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/topk_analysis/cache_same_area_full.json | /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/topk_analysis/posterior_k4.json
- `soft_orientation_target.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/teacher_signal_example_20260414/teacher_pair_soft_orientation_target_v2_smooth.png
- `useful_angle_set_module.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/vop_shortpaper_20260415_review/fig04a_angle_surface_mechanism.png
- `sparse_no_vop_match_visual.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
- `ours_vop_guided_match_visual.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png | best angle +20
- `match_visual_comparison_dual.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/drone/images/03_0079.JPG | /home/lcy/Workplace/GTA-UAV/Game4Loc/data/UAV_VisLoc_dataset/satellite/03_6_010_003.png
- `final_localization_result.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/pair_vop_assets_20260414/pair01_vop_summary.json
- `accuracy_efficiency_tradeoff_plot.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/vop_shortpaper_20260411/fig02b_dis1_runtime_tradeoff_sp_lg_frontier.png
- `localization_error_curve.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/figures/teacher_signal_example_20260414/teacher_pair_localization_error_curve.png

## Redrawn / Vectorized Modules

- `feature_concatenation_fusion_symbol.png` <- redrawn vector icon matched to the paper palette
- `topk_selected_angles.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/vop/topk_analysis/posterior_k4.json
- `pair_confidence_weight_curve.png` <- /home/lcy/Workplace/GTA-UAV/Game4Loc/train_vop.py
- `uav_topview_icon.png` <- redrawn vector icon because no local UAV icon asset exists
- `red_pin_icon.png` <- redrawn vector icon matched to the paper palette
- `arrow_right_icon.png` <- redrawn vector icon matched to the paper palette
- `rotation_arrow_icon.png` <- redrawn vector icon matched to the paper palette
- `topk_badge.png` <- redrawn vector badge
- `softmax_badge.png` <- redrawn vector badge

## Missing Inputs / Notes

- No critical source files were missing. No PPT/PPTX source deck was found; SVG drafts and figure PNGs were used instead.
- No local drone icon asset was found, so a vectorized UAV top-view icon was generated to match the paper style.
