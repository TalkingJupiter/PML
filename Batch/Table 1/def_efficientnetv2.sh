#!/bin/bash
#SBATCH --job-name=effnet-lightning
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/effnet_lightning/%x-%j.out
#SBATCH --error=slurm_logs/effnet_lightning/%x-%j.err

set -eo pipefail

WORKDIR="/mnt/DISCL/work/bsencer/PML"
ENVNAME="pmlcuda"

# ---- your script name (edit if different) ----
SCRIPT="train_efficientnetv2.py"

mkdir -p slurm_logs/effnet_lightning

echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Host  : $(hostname)"
echo "Date  : $(date)"
echo "PWD   : $(pwd)"
echo "====================================="

source ~/.bashrc

# Avoid "unbound variable" issues from conda activate.d scripts (MKL stuff)
set +u
conda activate "$ENVNAME"
set -u 2>/dev/null || true

echo "Python: $(which python)"
python -V

echo "Torch check:"
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "nvidia-smi:"
nvidia-smi || true

cd "$WORKDIR"

echo "Listing script:"
ls -lh "$SCRIPT"

echo "Running training now..."
python -u "$SCRIPT" --outdir experiments --run_name efficientnetv2

echo "TRAIN EXIT CODE: $?"