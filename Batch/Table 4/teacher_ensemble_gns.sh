#!/bin/bash 
#SBATCH --job-name=ensemble_teach_gns_2.8x
#SBATCH --partition=h100
#SBATCH --gres=gpu:1 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/Table4/%x-%A_%a.out
#SBATCH --error=slurm_logs/Table4/%x-%A_%a.err

# WIDTHS=(1.6 1.6 1.6)
# NOTE: For some reason the array job submission break the experiment. So I submitted manually
WIDTH=2.8
#${WIDTHS[$SLURM_ARRAY_TASK_ID]}
source ~/.bashrc
conda activate pmlcuda

RUN_NAME="ensemble_teacher_gns${WIDTH}x"

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


python training_with_teacher_ensemble_small.py \
  --student_width "${WIDTH}" \
  --outdir experiments \
  --run_name "${RUN_NAME}"