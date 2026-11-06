#!/bin/bash
#SBATCH --job-name=m10_dev
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a40|l40s|a100|v100"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=/project2/xiangren_1987/group_40/runs/logs/m10_dev_%j.out
#
# M10: run prompted claim baselines on the frozen DEV split and score them.
# Submit from the repo root:  sbatch scripts/slurm/m10_dev.sbatch
# Edit PROMPTS to pick which variants to run (each ~1-2 min after model load).

set -euo pipefail
module load conda
conda activate /project2/xiangren_1987/group_40/envs/nlp
export HF_HOME=/project2/xiangren_1987/group_40/hf_cache
cd /project2/xiangren_1987/group_40/NLP-Models-for-Scientific-Peer-Review

RUNS=/project2/xiangren_1987/group_40/runs
mkdir -p "$RUNS" results data/claims/predictions
nvidia-smi --query-gpu=name,memory.total --format=csv

PROMPTS=${PROMPTS:-"v1 v2"}
for P in $PROMPTS; do
  echo "=== prompt $P on dev ==="
  python scripts/baseline_claims.py --split dev --prompt "$P" \
    --out "data/claims/predictions/baseline_qwen7b_dev_${P}.jsonl" \
    --raw-out "$RUNS/m10_dev_${P}_raw.jsonl"
  python scripts/score_predictions.py claims \
    --pred "data/claims/predictions/baseline_qwen7b_dev_${P}.jsonl" \
    --gold data/claims/splits/dev.jsonl \
    --out "results/m10_dev_${P}.json"
done
echo "=== done ==="
