#!/bin/bash
#SBATCH --job-name=kd_gns2.8x-EfficienetV2
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/Table2/kd_gns2.8x-EfficienetV2/%x-%j.out
#SBATCH --error=slurm_logs/Table2/kd_gns2.8x-EfficienetV2/%x-%j.err

set -eo pipefail

WORKDIR="/mnt/DISCL/work/bsencer/PML"
ENVNAME="pmlcuda"


source ~/.bashrc
set +u
conda activate "$ENVNAME"
set -u 2>/dev/null || true

cd "$WORKDIR"

echo "Running training now..."
python training_with_kd_efficientnetv2_small.py \
  --outdir experiments \
  --run_name KD_GN-S_2.8x_from_effnetv2 \
  --student_width 2.8 \
  --teacher_run efficientnetv2_34083_602656884/checkpoints/\
  --teacher_ckpt last.ckpt \

# echo "TRAIN EXIT CODE: $?"