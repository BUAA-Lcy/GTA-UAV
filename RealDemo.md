# RealDemo.md

This file is the handoff for the real DJI-video demo work.

Read this before continuing the `Aerial-to-ground1/2/3` demo experiments. The
goal of this track is a fast, visually clear mentor-facing demo, not a formal
paper benchmark.


# 1. Scope And Intent

Current RealDemo goal:

- show that the input is a real UAV video captured by us;
- show satellite tile selection along the flight;
- show local matching correspondences in a clear panel;
- show a stable trajectory inset;
- show per-frame localized latitude / longitude;
- keep this demo path isolated from the paper/evaluator path as much as
  possible.

Important boundary:

- This demo currently does **not** perform real retrieval for the displayed
  Top-1.
- The displayed Top-1 satellite tile is selected by flight-log GPS nearest
  tile.
- This is intentional because the current request was to make a good-looking
  real-video demo quickly.
- Do not report these demo numbers as retrieval or localization benchmark
  results.


# 2. Current Best Demo Outputs

## Aerial-to-ground1

Dataset:

- `Game4Loc/data/Aerial-to-ground1/`
- Main source video used:
  - `dji_fly_20260420_182516_23_1776681926283_video.mp4`

Best archived DKM dense demo:

- MP4:
  - `Game4Loc/work_dir/aerial_ground1_demo/aerial_ground1_fullvideo_pose_top1_20260429_093345/pose_top1_dense_dkm_yaw_fullvideo_180f_direct_final/aerial_ground1_demo_video.mp4`
- Manifest:
  - `Game4Loc/work_dir/aerial_ground1_demo/aerial_ground1_fullvideo_pose_top1_20260429_093345/pose_top1_dense_dkm_yaw_fullvideo_180f_direct_final/demo_video_manifest.json`

Summary:

- duration: `60.06s`
- resolution: `1920x1080`
- rendered frames: `1800`
- sampled query cards: `180`
- matching backend: `dense_dkm`
- mean / min / max DKM inliers: `1557.46 / 65 / 4687`
- Top-1 satellite tile changes: `24`
- visual localization rejection counts:
  - `unavailable: 57`
  - `too_few_inliers: 31`
  - `gps_residual_gate: 21`

Earlier good sparse version, keep as archive:

- `Game4Loc/work_dir/aerial_ground1_demo/aerial_ground1_fullvideo_pose_top1_20260429_093345/pose_top1_sparse_yaw_fullvideo_180f_final/aerial_ground1_demo_video.mp4`


## Aerial-to-ground2

Dataset:

- `Game4Loc/data/Aerial-to-ground2/`
- Main source video used:
  - `dji_fly_20260420_185426_25_1776683433584_video.mp4`
- Flight log:
  - `DJIFlightRecord_2026-04-20_[18-48-40].csv`

Current corrected DKM dense demo:

- MP4:
  - `Game4Loc/work_dir/aerial_ground2_demo/aerial_ground2_fullvideo_pose_top1_20260429_121026/pose_top1_dense_dkm_yaw_square_query_aligned_match_visual_traj_180f_final/aerial_ground2_square_query_aligned_match_visual_traj_demo_video.mp4`
- Manifest:
  - `Game4Loc/work_dir/aerial_ground2_demo/aerial_ground2_fullvideo_pose_top1_20260429_121026/pose_top1_dense_dkm_yaw_square_query_aligned_match_visual_traj_180f_final/demo_video_manifest.json`

Summary:

- duration: `60.06s`
- resolution: `1920x1080`
- rendered frames: `1800`
- sampled query cards: `180`
- matching backend: `dense_dkm`
- query preprocessing: `center_square_crop_before_yaw_alignment_and_matching`
- match display: `yaw_aligned_query_coordinates`
- mean / min / max DKM inliers: `1242.31 / 55 / 4520`
- mean retained matches: `4941.70`
- accepted visual positions: `45`
- Top-1 satellite tile changes: `23`
- visual localization rejection counts:
  - `gps_residual_gate: 34`
  - `too_few_inliers: 37`
  - `unavailable: 64`
- satellite cache:
  - `Game4Loc/work_dir/aerial_ground2_demo/cache/gallery_z18/`
  - `252` z18 tiles were prepared for this run.

Current multi-scale SP+LG version:

- MP4:
  - `Game4Loc/work_dir/aerial_ground2_demo/aerial_ground2_fullvideo_pose_top1_20260429_121026/pose_top1_sparse_sp_lg_multiscale_yaw_square_query_aligned_visual_traj_180f_final/aerial_ground2_sparse_sp_lg_multiscale_square_query_aligned_visual_traj_demo_video.mp4`
- Manifest:
  - `Game4Loc/work_dir/aerial_ground2_demo/aerial_ground2_fullvideo_pose_top1_20260429_121026/pose_top1_sparse_sp_lg_multiscale_yaw_square_query_aligned_visual_traj_180f_final/demo_video_manifest.json`

Summary:

- duration: `60.06s`
- resolution: `1920x1080`
- rendered frames: `1800`
- sampled query cards: `180`
- matching backend: `sparse_sp_lg`
- multi-scale: `True`
- sparse scales: `[1.0, 0.8, 0.6, 1.2]`
- query preprocessing: `center_square_crop_before_yaw_alignment_and_matching`
- match display: `yaw_aligned_query_coordinates`
- mean / min / max inliers: `17.66 / 2 / 29`
- mean / min / max retained matches: `173.48 / 86 / 316`
- accepted visual positions: `52`
- Top-1 satellite tile changes: `23`
- visual localization rejection counts:
  - `too_few_inliers: 117`
  - `gps_residual_gate: 11`

Practical reading:

- The corrected DKM dense demo is the current preferred visual version because
  it has many more green correspondence lines.
- The multi-scale SP+LG demo is useful when the user specifically asks for a
  sparse-feature version, but the correspondence panel is much sparser.
- Aerial-to-ground2 is harder than Aerial-to-ground1 because it contains more
  repeated farmland textures and weaker distinctive structures.

Superseded Aerial-to-ground2 archives:

- `pose_top1_dense_dkm_yaw_fullvideo_180f_final/aerial_ground2_demo_video.mp4`
  - old rectangular query version.
- `pose_top1_dense_dkm_yaw_square_query_visual_traj_180f_final/aerial_ground2_square_query_visual_traj_demo_video.mp4`
  - square query version, but match points were still displayed on the
    unrotated query image. Do not use this version for reporting.


## Aerial-to-ground3

Dataset:

- `Game4Loc/data/Aerial-to-ground3/`
- Videos:
  - `dji_fly_20260423_101228_16_1776914613676_video.mp4`
  - `dji_fly_20260423_101500_17_1776913644134_video.mp4`
- Flight log:
  - `DJIFlightRecord_2026-04-23_[10-50-33].csv`

Current status:

- Data has been inspected and recorded in the experimental report.
- Aerial-to-ground3 has not yet been run through the RealDemo renderer.
- Treat it as future work unless the user explicitly asks to generate a new
  demo from it.


# 3. Current Demo Pipeline

The current presentation pipeline is:

1. Parse DJI flight log CSV.
2. Decode frames from the selected DJI MP4.
3. Sample frames across the source video, avoiding exact first and last frames.
4. Center-crop the UAV query frame to a square before matching / display.
5. Download / reuse satellite tiles at zoom `18` around the flight track.
6. For each sampled frame, choose displayed Top-1 by nearest satellite tile to
   flight-log latitude / longitude.
7. Align query and satellite orientation using UAV yaw metadata.
8. Run dense DKM or multi-scale SP+LG local matching.
9. Draw only green inlier correspondence lines.
10. Estimate a visual center projection from the homography when quality gates
    pass.
11. Render cards and encode a slow MP4.

The display board currently contains:

- top-left: real UAV frame, square-cropped in the latest Aerial-to-ground2 runs;
- top-middle: flight-pose selected Top-1 satellite tile;
- top-right: target visualization on local satellite crop;
- bottom: green inlier correspondences;
- right-center inset in the bottom panel: trajectory comparison;
- bottom text: query GPS, localized latitude / longitude, and Top-1 tile name.

Technical text intentionally removed from the visual panel:

- no `DKM dense` label;
- no `selected inlier lines (...)` line;
- no `Dis` value;
- no cross marker in Top-1 retrieval view;
- no `Top-1 source: flight GPS nearest tile` line.


# 4. Important Code Paths

Main demo renderer:

- `Game4Loc/scripts/render_aerial_ground1_demo_video.py`

Important renderer behavior:

- supports `--top1_selection_mode flight_pose_top1`;
- changes Top-1 title to `Flight-pose top-1 tile`;
- writes `Localized: (lat, lon)`;
- can use dense DKM or sparse SP+LG;
- supports square query preprocessing;
- supports yaw-aligned match display;
- can disable Top-1-center fallback for displayed visual localization;
- rejects poor visual localization using:
  - `visual_calc_min_inliers = 300`
  - `visual_calc_min_inlier_ratio = 0.04`
  - `visual_calc_max_gps_error_m = 80.0`

Recent helper functions added to the renderer:

- `rotate_bgr_with_affine(image_bgr, angle_deg)`
- `warp_points_affine(points_xy, affine_mat)`
- `prepare_aligned_query_match_debug(query_bgr, debug, rot_angle)`

These helpers are important because the latest display panel shows feature
correspondences on the yaw-rotated query image, not on the original unrotated
query image.

Support scripts:

- `Game4Loc/scripts/run_aerial_ground1_demo.py`
- `Game4Loc/scripts/run_aerial_ground1_continuous_demo.py`

These scripts were originally named for `Aerial-to-ground1`, but most helper
functions are reusable for `Aerial-to-ground2`.

Dense matcher:

- `Game4Loc/game4loc/matcher/gim_dkm.py`

Sparse matcher:

- `Game4Loc/game4loc/matcher/sparse_sp_lg.py`

Current DKM-specific note:

- The dense DKM path was patched to use direct Homography-RANSAC for the demo
  path because the older fundamental-matrix prefilter could return all-zero
  match summaries on these real videos.
- For visual center projection, the dense matcher homography is effectively
  gallery-to-query, so the demo projection uses `inv(H)` to project the query
  center back into the satellite tile.
- This is important. If future agents see localization jumping or always
  unavailable, check this homography direction first.

Current SP+LG-specific note:

- The sparse SP+LG homography path is query-to-gallery in the current demo
  projection logic.
- Do not copy the DKM `inv(H)` projection rule into the sparse path without
  checking the stored homography direction.


# 5. The Critical Visualization Fix

Problem that was observed:

- The user correctly noticed that the feature-point mapping drawn on the board
  was still wrong.
- Internally, matching used yaw alignment, but the visualization drew query-side
  match points on the original unrotated query image.
- This made correspondence lines appear geometrically inconsistent, even when
  the matcher / homography output itself was usable.

Correct behavior:

- Center-crop query to square first.
- Rotate the square query by yaw for matching / display.
- Transform query-side match points through the same forward rotation affine.
- Draw the bottom correspondence panel using:
  - yaw-aligned query image on the left;
  - satellite tile on the right;
  - transformed query points;
  - original satellite points.

Expected visual artifact:

- The yaw-rotated square query may appear as a diamond-like image with black
  triangular padding. This is expected because the rotated canvas expands to
  preserve the image content.

Do not regress this:

- If feature lines look wrong again, first check whether
  `match_display = yaw_aligned_query_coordinates` is present in the manifest.
- Also inspect q060 / q150 cards from the latest Aerial-to-ground2 output; they
  are good reference examples for the corrected display convention.


# 6. Reproducing The Current Demos

Use the same environment:

- `/home/lcy/miniconda3/envs/gtauav/bin/python`

The current full-video pose-top1 run directories were produced by a one-off
Python driver that reused helper functions from:

- `scripts/run_aerial_ground1_demo.py`
- `scripts/render_aerial_ground1_demo_video.py`

For future reproducibility, the next agent should promote that one-off driver
into a real script, for example:

- `Game4Loc/scripts/run_realvideo_pose_top1_demo.py`

Recommended CLI shape:

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/scripts/run_realvideo_pose_top1_demo.py \
  --data_root Game4Loc/data/Aerial-to-ground2 \
  --target_video Game4Loc/data/Aerial-to-ground2/dji_fly_20260420_185426_25_1776683433584_video.mp4 \
  --output_root Game4Loc/work_dir/aerial_ground2_demo \
  --zoom 18 \
  --sample_max_frames 180 \
  --top1_mode flight_pose_top1 \
  --match_backend dense_dkm \
  --square_query_crop \
  --yaw_align \
  --aligned_match_display \
  --video_fps 29.97 \
  --hold_frames_per_card 10
```

Until that script exists, continue by copying the existing one-off logic from
the current implementation pattern:

- build `results.json` with `topk_paths[0]` set to the nearest flight-GPS tile;
- set `display_top1_selection_mode = "flight_pose_top1"`;
- center-crop query frames to square;
- align query / satellite orientation with yaw;
- render match panels in yaw-aligned query coordinates;
- write card JPGs and MP4;
- save `demo_video_manifest.json`.


# 7. Satellite Tile Handling

Tile source currently used:

- ArcGIS World Imagery:
  - `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}`

Current zoom:

- `z18`

Cache locations:

- Aerial-to-ground1:
  - `Game4Loc/work_dir/aerial_ground1_demo/cache/gallery_z18/`
- Aerial-to-ground2:
  - `Game4Loc/work_dir/aerial_ground2_demo/cache/gallery_z18/`

Useful behavior:

- It is safe to copy overlapping z18 tiles from one demo cache into another
  cache before downloading missing tiles.
- Do not force redownload unless imagery is clearly corrupted or placeholder.
- If satellite imagery looks blurry, consider preparing a higher-zoom display
  cache, but keep retrieval / tile selection bookkeeping explicit.

Known imagery issue:

- Some z18 tiles are not extremely sharp.
- Higher zoom might improve visual appeal, but it requires careful handling of
  tile coordinate conversion and larger cache size.


# 8. Matching And Visualization Notes

Current preferred visual matcher for the real demo:

- dense DKM, green inlier lines only.

Use SP+LG when the user explicitly asks for sparse / multi-scale matching:

- multi-scale enabled;
- current scales: `[1.0, 0.8, 0.6, 1.2]`;
- lines are clearer but much fewer than DKM.

Useful current DKM display settings:

- `dkm_top_conf_lines = 900`
- `dkm_line_thickness = 1`
- `rotate = 360.0`
- `yaw0 = 0.0`
- `yaw1 = -select_alignment_yaw(result, "auto")`

Quality gates for using visual center projection:

- reject if DKM inliers `< 300`;
- reject if inlier ratio `< 0.04`;
- reject if projected visual coordinate is `> 80m` from flight-log GPS;
- otherwise accept visual center as `dense_dkm_homography_center`.

Trajectory inset:

- The latest Aerial-to-ground2 corrected demos show the real estimated visual
  positions after quality gates.
- The inset has been moved from bottom-right to the right-center of the bottom
  match panel.
- Draw only `results[:current_pos+1]`; never draw future trajectory points in
  the current frame.
- Orange should represent the flight-log / GPS path.
- Green should represent the current estimated visual/localized path.


# 9. Known Failure Modes

## DKM all-zero matches

Symptom:

- every frame has `0` inliers;
- match panel says no valid dense inlier set.

Likely causes:

- using the old dense DKM geometric prefilter;
- using a stale code path that calls the generic `run_match_visuals` path in a
  way that does not preserve the direct homography result;
- GPU / WSL memory instability after long DKM batches.

Recommended fix:

- use direct per-frame DKM invocation;
- check `matcher.get_last_match_info()`;
- check `matcher.get_last_match_debug()`;
- ensure dense `last_match_debug["homography"]` is present;
- call `torch.cuda.empty_cache()` every few frames.

## Feature lines look geometrically wrong

Symptom:

- the matcher produces nonzero inliers;
- localization may look plausible;
- but the green lines drawn on the board do not correspond to the visible
  query image geometry.

Likely cause:

- the query image was yaw-rotated for matching, but the line endpoints were
  drawn on the original unrotated query image.

Recommended fix:

- display the yaw-rotated query image in the match panel;
- transform query-side match coordinates with the same forward affine;
- keep satellite-side points unchanged;
- verify the manifest says `match_display: yaw_aligned_query_coordinates`.

## Wrong localized coordinate

Symptom:

- match lines look plausible but localized lat/lon is far from GPS;
- trajectory jumps.

Check:

- homography direction;
- for DKM dense, projection should use `inv(H)` for query-center-to-gallery;
- for SP+LG, projection currently uses the stored query-to-gallery homography;
- quality gates should reject large GPS residuals.

## Right-bottom trajectory messy at frame 1

Symptom:

- green trajectory appears as a full messy path immediately.

Fix:

- the inset must draw only `results[:current_pos+1]`;
- do not project all future predicted points into the current frame;
- do not fall back to Top-1 center when the visual estimate is unavailable if
  the goal is to show the real estimated trajectory.

## Repeated farmland false correspondences

Symptom:

- lines cluster in repeated crop-row textures;
- inlier count may be nonzero but localization is not trustworthy.

Fix:

- keep GPS residual gate;
- do not trust raw inlier count alone;
- consider selecting a better segment with stronger structures such as roads,
  buildings, intersections, or field boundaries.


# 10. Experimental Report

The user provided a `.docx` template and asked for a report based on the real
flight experiments.

Original template:

- `Game4Loc/data/无人机视觉自主导航系统飞行实验报告_后验证版.docx`

Generated report:

- `Game4Loc/data/无人机视觉自主导航系统飞行实验报告_RealDemo_后验证版.docx`

Report scope:

- Aerial-to-ground1 and Aerial-to-ground2 are described as completed demo
  datasets.
- Aerial-to-ground3 is described as collected / inspected, not yet rendered as
  a RealDemo output.
- The report intentionally avoids claiming formal benchmark performance from
  the demo videos.


# 11. Recommended Next Experiments

Priority 1: formalize the one-off runner.

- Add `Game4Loc/scripts/run_realvideo_pose_top1_demo.py`.
- It should accept arbitrary `data_root`, `target_video`, `output_root`, and
  output filename prefix.
- It should reproduce the current Aerial-to-ground1 and Aerial-to-ground2
  demos without copy-pasting terminal snippets.

Priority 2: segment selection for best-looking demo.

- Use full-video manifests to find windows with high mean inliers and frequent
  Top-1 tile changes.
- Generate a shorter high-quality clip if the user wants a prettier version.
- Keep the full 60s version as evidence that the method runs across the whole
  video.

Priority 3: improve satellite visual quality.

- Try z19 display-only satellite tiles.
- Keep z18 metadata / nearest-tile selection unchanged unless the full pipeline
  is updated carefully.
- Validate that target crop and Top-1 tile marker still align after zoom
  conversion.

Priority 4: compare DKM and sparse versions only visually.

- Do not turn the real demo into a formal benchmark.
- If comparing, compare:
  - visual readability;
  - match stability;
  - line density;
  - trajectory smoothness;
  - speed to generate.

Priority 5: make output naming dataset-neutral.

- Current scripts and titles still contain `aerial_ground1` names.
- For future demos, rename output labels to `realvideo_demo` or
  `aerial_ground_demo` to avoid confusion.

Priority 6: run Aerial-to-ground3 only if requested.

- Aerial-to-ground3 has lower-looking pitch angle ranges in the inspected log
  and may be visually different from Aerial-to-ground2.
- Do not start long new rendering jobs unless the user asks for them.


# 12. What Not To Do

Do not:

- claim this is real retrieval performance;
- report `flight_pose_top1` as retrieval Top-1;
- mix these demo outputs into paper tables;
- remove the GPS residual gate just to show more visual points;
- trust inlier count alone as correctness;
- silently change DJI yaw / gimbal yaw metadata semantics;
- draw yaw-aligned match points on an unrotated query image;
- rewrite the main evaluator path for this demo;
- force changes into official GTA / VisLoc evaluation unless explicitly asked;
- delete the existing good demo outputs.


# 13. Quick Status For Future Agents

Current user preference:

- make a visually strong demo first;
- keep the main board format stable;
- use flight latitude / longitude to select satellite Top-1 for the demo;
- show localized latitude / longitude;
- use DKM dense correspondences when a visually rich board is preferred;
- use multi-scale SP+LG when the user explicitly requests sparse matching;
- crop query frames to square before matching / display;
- yaw-align the query image and satellite before drawing correspondences;
- keep lines green and visually readable;
- keep trajectory stable and honest about accepted visual estimates.

Current best recommendation:

- For Aerial-to-ground2, use the corrected DKM dense square-query aligned
  version:
  - `Game4Loc/work_dir/aerial_ground2_demo/aerial_ground2_fullvideo_pose_top1_20260429_121026/pose_top1_dense_dkm_yaw_square_query_aligned_match_visual_traj_180f_final/aerial_ground2_square_query_aligned_match_visual_traj_demo_video.mp4`
- For sparse comparison, use the multi-scale SP+LG version:
  - `Game4Loc/work_dir/aerial_ground2_demo/aerial_ground2_fullvideo_pose_top1_20260429_121026/pose_top1_sparse_sp_lg_multiscale_yaw_square_query_aligned_visual_traj_180f_final/aerial_ground2_sparse_sp_lg_multiscale_square_query_aligned_visual_traj_demo_video.mp4`
- For a cleaner first mentor-facing video, Aerial-to-ground1 DKM is still a
  strong option:
  - `Game4Loc/work_dir/aerial_ground1_demo/aerial_ground1_fullvideo_pose_top1_20260429_093345/pose_top1_dense_dkm_yaw_fullvideo_180f_direct_final/aerial_ground1_demo_video.mp4`
