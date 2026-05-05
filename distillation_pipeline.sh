#!/bin/bash
#SBATCH --job-name=distill_and_plot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --exclusive
#SBATCH --partition=h100
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# 1. Setup environment
mkdir -p slurm_logs
mkdir -p plots

# 2. Run the distillation experiments
# master_distillation_loop.py will automatically:
#  - Detect SLURM and run for 200 epochs
#  - Discover local/HF teachers
#  - Perform distillation strictly on internal features
#  - Log metrics to history.json
echo "Starting distillation experiments..."
uv run master_distillation_loop.py

# 3. Final verification of plots
echo "Ensuring all plots are generated..."
uv run plots/distillation_analysis.py

echo "Pipeline complete. Check the 'plots/' directory for results."
