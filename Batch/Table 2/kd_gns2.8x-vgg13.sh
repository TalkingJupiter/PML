#!/bin/bash
#SBATCH --job-name=kd_gns2.8x-vgg13
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/Table2/kd_gns2.8x-vgg13/%x-%j.out
#SBATCH --error=slurm_logs/Table2/kd_gns2.8x-vgg13/%x-%j.err

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

echo "Running training now..."
python training_with_kd_vgg13_bn_small.py \
  --outdir experiments \
  --run_name KD_GN-S_2.8x_from_VGG13BN \
  --student_width 2.8 \
  --teacher_run  vgg13/vgg13_bn_seed0 \
  --teacher_ckpt best_model.pth \
  --temperature 1.0 \
  --alpha 0.5

# echo "TRAIN EXIT CODE: $?"