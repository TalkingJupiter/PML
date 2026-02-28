#!/bin/bash
#SBATCH --job-name=gns_sweep
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

# set -euo pipefail

# ---- Define width list ----
WIDTHS=(1.0 1.3 1.6 1.9 2.2 2.8 3.4)

WIDTH=${WIDTHS[$SLURM_ARRAY_TASK_ID]}

# ---- Paths ----
WORKDIR="/mnt/DISCL/work/bsencer/PML"
OUTDIR="/mnt/DISCL/work/bsencer/PML/experiments"
ENVNAME="pmlcuda"

RUN_NAME="gns_${WIDTH}x_seed0"

mkdir -p slurm_logs
mkdir -p "${OUTDIR}"

echo "=== Job Info ==="
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Width: $WIDTH"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "================"

# ---- Environment ----
source ~/.bashrc
conda activate "${ENVNAME}"

# ---- Sanity check ----
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
nvidia-smi || true

# ---- Run ----
cd "${WORKDIR}"

python train_default_small.py \
  --width "${WIDTH}" \
  --outdir "${OUTDIR}" \
  --run_name "${RUN_NAME}"