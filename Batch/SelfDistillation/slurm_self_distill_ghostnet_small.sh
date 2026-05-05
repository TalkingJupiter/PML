#!/usr/bin/env bash
#SBATCH --job-name=gns_sd_all
#SBATCH --partition=h100
#SBATCH --reservation=cpufreq
#SBATCH --nodes=1
#SBATCH --nodelist=rpg-93-[3-4]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Make sure this exists before sbatch submission in practice
mkdir -p logs/slurm

source /mnt/DISCL/home/bsencer/miniforge3/etc/profile.d/conda.sh
set +u
conda activate pmlcuda
set -u

WIDTHS=(1.0)
WIDTH="${WIDTHS[$SLURM_ARRAY_TASK_ID]}"

OUTDIR="${OUTDIR:-experiments}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TEMPERATURE="${TEMPERATURE:-5.0}"
ALPHA="${ALPHA:-0.7}"
SCRIPT_PATH="${SCRIPT_PATH:-training_with_self_distillation_ghostnet_small.py}"

TEACHER_RUN="gns_${WIDTH}x_seed0"
TEACHER_CKPT="best_model.pth"
RUN_NAME="sd_gns_${WIDTH}x_seed0"

TEACHER_PATH="${OUTDIR}/${TEACHER_RUN}/${TEACHER_CKPT}"
if [[ ! -f "${TEACHER_PATH}" ]]; then
  echo "[ERROR] Teacher checkpoint not found: ${TEACHER_PATH}"
  exit 1
fi

echo "===================================================="
echo " Offline Self-Distillation: GhostNetV3 Small"
echo "===================================================="
echo "[INFO] Width       : ${WIDTH}"
echo "[INFO] Teacher run : ${TEACHER_RUN}"
echo "[INFO] Teacher ckpt: ${TEACHER_PATH}"
echo "[INFO] Run name    : ${RUN_NAME}"
echo "[INFO] Temp        : ${TEMPERATURE}"
echo "[INFO] Alpha       : ${ALPHA}"
echo "[INFO] Partition   : h100"
echo "[INFO] Host        : $(hostname)"
echo "[INFO] Start       : $(date)"
echo "===================================================="

"${PYTHON_BIN}" "${SCRIPT_PATH}" \
  --width "${WIDTH}" \
  --outdir "${OUTDIR}" \
  --run_name "${RUN_NAME}" \
  --teacher_run "${TEACHER_RUN}" \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --temperature "${TEMPERATURE}" \
  --alpha "${ALPHA}"

STATUS=$?
echo "[INFO] End       : $(date)"
echo "[INFO] Exit code : ${STATUS}"
exit ${STATUS}