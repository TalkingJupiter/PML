#!/bin/bash
#SBATCH --job-name=densenet161-base
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/densenet161/%x-%j.out
#SBATCH --error=slurm_logs/densenet161/%x-%j.err

set -eo pipefail
# set -x

WORKDIR="/mnt/DISCL/work/bsencer/PML"
ENVNAME="pmlcuda"
OUTDIR="/mnt/DISCL/work/bsencer/PML/experiments/densenet"

mkdir -p slurm_logs/densenet161
mkdir -p "$OUTDIR"

source ~/.bashrc
set +u
conda activate "$ENVNAME"
set -u 2>/dev/null || true

cd "$WORKDIR"

# echo "Python: $(which python)"
# python -V

# echo "Listing training script:"
# ls -lh train_densenet.py

# # echo "Argparse help output:"
# # python -u train_densenet.py --help || true

echo "Running training now..."
python -u train_densenet.py \
  --model densenet161 \
  --outdir "$OUTDIR"

# echo "TRAIN EXIT CODE: $?"