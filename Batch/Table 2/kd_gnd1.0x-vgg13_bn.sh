#!/bin/bash
#SBATCH --job-name=kd_gnd1.0x-vgg13_bn
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/Table2/kd_gnd1.0x-vgg13_bn/%x-%j.out
#SBATCH --error=slurm_logs/Table2/kd_gnd1.0x-vgg13_bn/%x-%j.err

set -eo pipefail
# set -x

WORKDIR="/mnt/DISCL/work/bsencer/PML"
ENVNAME="pmlcuda"
OUTDIR="/mnt/DISCL/work/bsencer/PML/experiments/Table2"

mkdir -p slurm_logs/Table2/kd_gnd1.0x-vgg13_bn
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
python training_with_kd_vgg13_bn.py \
  --outdir experiments \
  --run_name KD_GN-D_1.0x_from_vgg13_bn \
  --student_width 1.0 \
  --teacher_run vgg13/vgg13_bn_seed0 \
  --teacher_ckpt best_model.pth

# echo "TRAIN EXIT CODE: $?"