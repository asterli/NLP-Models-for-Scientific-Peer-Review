#!/bin/bash
#SBATCH --job-name=m11_train
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a40|l40s|a100"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/project2/xiangren_1987/group_40/runs/logs/m11_train_%j.out
#
# M11: LoRA fine-tune the claim extractor on the frozen TRAIN split, select on DEV, then
# re-score the best adapter from a fresh process so the reload path is verified.
#   sbatch scripts/slurm/m11_train.sbatch                      # defaults
#   RUN=r8_lr1e-4_s544 EXTRA="--lora-r 8 --lr 1e-4" sbatch scripts/slurm/m11_train.sbatch
# bf16 GPUs only (A40/L40S/A100). On a V100 add EXTRA="--load-4bit" and drop the constraint.

set -euo pipefail
module load conda
conda activate /project2/xiangren_1987/group_40/envs/nlp
export HF_HOME=/project2/xiangren_1987/group_40/hf_cache
cd /project2/xiangren_1987/group_40/NLP-Models-for-Scientific-Peer-Review

RUN=${RUN:-r16_lr2e-4_s544}
EXTRA=${EXTRA:-}
OUT=/project2/xiangren_1987/group_40/runs/m11/$RUN
mkdir -p "$OUT" results data/claims/predictions
nvidia-smi --query-gpu=name,memory.total --format=csv

python scripts/train_claims.py --out "$OUT" $EXTRA

# Fresh-process reload check + dev score with the team scorer (same numbers as in summary.json).
python scripts/predict_claims.py --adapter "$OUT/best" --split dev \
  --out "data/claims/predictions/finetuned_dev_${RUN}.jsonl" $EXTRA
python scripts/score_predictions.py claims \
  --pred "data/claims/predictions/finetuned_dev_${RUN}.jsonl" \
  --gold data/claims/splits/dev.jsonl --out "results/m11_dev_${RUN}.json"
echo "=== done: $OUT ==="
