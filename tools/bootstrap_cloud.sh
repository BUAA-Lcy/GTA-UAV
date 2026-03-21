#!/bin/bash
set -e

echo "==========================================================="
echo "       GTA-UAV Cloud Container Bootstrap Script"
echo "==========================================================="

# 1. Check Conda
if ! command -v conda &> /dev/null; then
    echo "[Error] conda could not be found. Please install miniconda or anaconda first."
    echo "For example:"
    echo "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

ENV_NAME="gtauav"

# 2. Create environment if it doesn't exist
if conda env list | grep -q "$ENV_NAME"; then
    echo "[Info] Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "[Info] Creating conda environment '$ENV_NAME' (python 3.10)..."
    conda create -n $ENV_NAME python=3.10 -y
fi

# 3. Initialize conda for bash script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 4. Install dependencies
echo "[Info] Installing requirements..."
# Install PyTorch specifically for CUDA (assuming CUDA 11.8 or 12.1 is available on cloud)
# This will usually override the pip version if a specific channel is needed, but for simplicity:
pip install -r Game4Loc/requirements.txt

# 5. Create necessary directories
echo "[Info] Creating necessary directories..."
mkdir -p logs
mkdir -p debug_outputs
mkdir -p Game4Loc/work_dir
mkdir -p Game4Loc/pretrained

echo "==========================================================="
echo " Bootstrap completed successfully!"
echo ""
echo " Next Steps:"
echo " 1. Make sure your dataset is placed correctly in: Game4Loc/data/GTA-UAV-data/"
echo " 2. Activate environment:  conda activate $ENV_NAME"
echo " 3. Run smoke test:        ./tools/run_smoke_test.sh"
echo " 4. Run formal experiment: ./tools/run_pose_gate_round2.sh"
echo "==========================================================="
