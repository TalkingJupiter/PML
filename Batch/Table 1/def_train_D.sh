#!/bin/bash
#SBATCH --job-name=gnd
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/D/%x-%j.out
#SBATCH --error=slurm_logs/D/%x-%j.err

# set -euo pipefail

# ---- Define width list ----


# ---- Paths ----
WORKDIR="/mnt/DISCL/work/bsencer/PML"
OUTDIR="/mnt/DISCL/work/bsencer/PML/experiments"
ENVNAME="pmlcuda"

RUN_NAME="gnd_1.0x_seed0"

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

python train_default.py \
  --outdir "${OUTDIR}" \
  --run_name "${RUN_NAME}"