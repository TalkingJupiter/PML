#!/bin/bash
#SBATCH --job-name=kd_gns1.0x-gns2.8x
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/Table2/kd_gns1.0x-gns2.8x/%x-%j.out
#SBATCH --error=slurm_logs/Table2/kd_gns1.0x-gns2.8x/%x-%j.err

set -eo pipefail
# set -x

WORKDIR="/mnt/DISCL/work/bsencer/PML"
ENVNAME="pmlcuda"




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

python training_with_kd_ghostnet_small.py \
  --outdir experiments \
  --run_name KD_GN-S_1.0x-GN-S_2.8x \
  --student_width 1.0 \
  --teacher_width 2.8 \
  --teacher_run gns_2.8x_seed0 \
  --teacher_ckpt best_model.pth \
  --temperature 5.0 \
  --alpha 0.7