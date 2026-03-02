#!/bin/bash 
#SBATCH --job-name=train_w_teach_gns
#SBATCH --partition=h100
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/Table3/%x-%A_%a.out
#SBATCH --error=slurm_logs/Table3/%x-%A_%a.err

WIDTHS=(1.0 1.3 1.6)
WIDTH=${WIDTHS[$SLURM_ARRAY_TASK_ID]}

ENVNAME="pmlcuda"

RUN_NAME="train_w_teacher_gns${WIDTH}x"

echo "=== Job Info ==="
echo "Array ID: $SLURM_ARRAY_TASK_ID"
echo "Width: $WIDTH"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "================"

# ---- Sanity check ----
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
nvidia-smi || true

# ---- Run ----


python training_with_teacher_assistant_small.py \
  --teacher_run experiments/resnet \
  --teacher_ckpt best_model.pth \
  --student_width "${WIDTH}" \
  --outdir experiments \
  --run_name "${RUN_NAME}"