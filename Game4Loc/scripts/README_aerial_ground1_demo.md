# Aerial-to-ground1 Demo

This is an isolated demo path for the raw `Aerial-to-ground1` drop.

It does **not** modify the existing retrieval training, fine-localization
evaluators, matcher internals, or the current paper pipeline. The only code
addition is the standalone script:

- `scripts/run_aerial_ground1_demo.py`

All outputs are written under:

- `work_dir/aerial_ground1_demo/`

The downloaded Laishui-area satellite gallery is cached separately under:

- `work_dir/aerial_ground1_demo/cache/gallery_z18/` by default

## 1. Prepare the satellite gallery only

```bash
source /home/lcy/miniconda3/etc/profile.d/conda.sh
conda activate gtauav
cd /home/lcy/Workplace/GTA-UAV/Game4Loc
python scripts/run_aerial_ground1_demo.py --download_only
```

## 2. Run the full mentor-facing demo

```bash
source /home/lcy/miniconda3/etc/profile.d/conda.sh
conda activate gtauav
cd /home/lcy/Workplace/GTA-UAV/Game4Loc
python scripts/run_aerial_ground1_demo.py \
  --sample_interval_sec 30 \
  --sample_max_frames 10 \
  --zoom 18
```

If you want a cleaner mentor demo that emphasizes more downward-looking frames,
add a gimbal-pitch filter:

```bash
python scripts/run_aerial_ground1_demo.py \
  --sample_max_frames 6 \
  --sample_start_sec 280 \
  --sample_end_sec 390 \
  --query_max_gimbal_pitch_deg -45 \
  --zoom 18
```

## 3. Artifacts

Each run creates a new timestamped folder with:

- `queries/`: sampled video frames
- `contact_sheet_topk.jpg`: query/top-k retrieval sheet
- `trajectory_top1_vs_gps.jpg`: GPS track vs predicted top-1 tile centers
- `summary.md`: concise report for sharing
- `summary.json`: machine-readable dump

## 4. Render an intuitive demo video

After a run is complete, render an MP4/GIF with:

- UAV frame
- retrieved top-1 tile
- target/localization visualization
- sparse match-line panel

```bash
source /home/lcy/miniconda3/etc/profile.d/conda.sh
conda activate gtauav
cd /home/lcy/Workplace/GTA-UAV/Game4Loc
python scripts/render_aerial_ground1_demo_video.py \
  --run_dir /home/lcy/Workplace/GTA-UAV/Game4Loc/work_dir/aerial_ground1_demo/<run_name>
```

The rendered assets are written to:

- `<run_dir>/demo_video_assets/`
- `aerial_ground1_demo_video.mp4`
- `aerial_ground1_demo_video.gif`
- `cards/`
- `match_vis/`

## 5. Important note

The script reports per-frame GPS errors as **proxy demo metrics**. Raw MP4
frames are aligned to the DJI flight log by sequential coverage over the
continuous `CAMERA.isVideo` segment, which is good enough for a fast demo but
should not be treated as a paper-grade evaluation protocol.
