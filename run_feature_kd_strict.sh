#!/bin/bash
#SBATCH --job-name=feature_kd_strict
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Ensure log directory exists
mkdir -p slurm_logs

# Run the master loop with the --strict flag
# This ensures training is strictly on internal teacher feature outputs
uv run master_distillation_loop.py --strict
