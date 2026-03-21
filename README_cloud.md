# GTA-UAV Cloud Deployment Guide

This project contains the implementation of the UAV-to-Satellite visual localization with Pose Attention Gating. The code is ready for single-machine, multi-GPU (e.g., 2x5090) cloud container deployment.

## 1. Quick Start (Bootstrap)

Upload the entire project directory (excluding `logs/` and `Game4Loc/data/`) to your cloud container.
Run the bootstrap script to automatically check conda, create the environment, and install dependencies:

```bash
chmod +x tools/bootstrap_cloud.sh
./tools/bootstrap_cloud.sh
```

## 2. Manual Environment Setup (Alternative)

If you prefer to set up manually:
```bash
conda create -n gtauav python=3.10 -y
conda activate gtauav
pip install -r Game4Loc/requirements.txt
```

## 3. Data Preparation

Ensure your dataset is placed at `Game4Loc/data/GTA-UAV-data/`.
The directory structure should look like this:
```text
Game4Loc/data/GTA-UAV-data/
  ├── cross-area-drone2sate-test.json
  ├── cross-area-drone2sate-train.json
  ├── ... (other metadata files)
  ├── drone/
  └── sate/
```
*Note: Do not upload the data folder via Git/SCP directly if it's too large. Download/mount it on the cloud instance directly.*

## 4. Pre-trained Weights

Timm will automatically download the required ViT backbone weights (e.g., `vit_base_patch16_rope_reg1_gap_256.sbb_in1k`) to your local cache (`~/.cache/huggingface/hub/` or `~/.cache/torch/hub/`) during the first run.
Ensure your cloud instance has internet access for the initial download, or manually upload the `.bin` / `.pth` files to the cache directory if it's an offline container.

## 5. Smoke Test

Before running long experiments, run the smoke test to verify dataloaders, environment, and paths:

```bash
conda activate gtauav
./tools/run_smoke_test.sh
```
If this completes successfully, the environment is perfectly healthy.

## 6. Formal Experiments

Run the Round 2 Pose Gate Ablation experiments:

```bash
conda activate gtauav
./tools/run_pose_gate_round2.sh
```

This will sequentially run 7 short-run experiments (3 epochs each) to verify the performance of different `lambda` and `floor` settings. All logs will be independently saved in `logs/` with a summarized report generated at the end.
