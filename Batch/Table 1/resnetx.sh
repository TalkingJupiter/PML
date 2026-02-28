#!/bin/bash
#SBATCH --job-name=resnetX
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/resnet/%x-%A_%a.out
#SBATCH --error=slurm_logs/resnet/%x-%A_%a.err

MODELS=(resnet18 resnet34 resnet50)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

WORKDIR="/mnt/DISCL/work/bsencer/PML"
OUTDIR="/mnt/DISCL/work/bsencer/PML/experiments/resnet"
ENVNAME="pmlcuda"

RUN_NAME="${MODEL}_seed0"

mkdir -p slurm_logs
mkdir -p "${OUTDIR}"

echo "=== Job Info ==="
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "================"

source ~/.bashrc
conda activate "${ENVNAME}"

# ---- Sanity check ----
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
nvidia-smi || true

# ---- Run ----
cd "${WORKDIR}"

python train_resnet.py \
    --model "${MODEL}" \
    --outdir "${OUTDIR}" \
    --run_name "${RUN_NAME}"