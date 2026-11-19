#!/bin/bash
#SBATCH --job-name=m12_test
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a40|l40s|a100"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --output=/project2/xiangren_1987/group_40/runs/logs/m12_test_%j.out
#
# M12: the ONE held-out TEST evaluation of the claim extractor.
#   prompted baseline (M10, prompt v2)  vs  fine-tuned LoRA adapter (M11, frozen run)
# Same test papers, same parser, same scorer. Settings were frozen on dev before this ran.
#   sbatch scripts/slurm/m12_test.sbatch
# Refuses to run if test results already exist, so test is not re-used by accident (FORCE=1 overrides).

set -euo pipefail
module load conda && eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/group_40/envs/nlp
export HF_HOME=/project2/xiangren_1987/group_40/hf_cache
cd /project2/xiangren_1987/group_40/NLP-Models-for-Scientific-Peer-Review

RUN=${RUN:-r16_lr2e-4_s544}
RUNS=/project2/xiangren_1987/group_40/runs
ADAPTER=$RUNS/m11/$RUN/best
BASE_PRED=data/claims/predictions/baseline_qwen7b_test_v2.jsonl
FT_PRED=data/claims/predictions/finetuned_test_${RUN}.jsonl
BASE_RES=results/m12_test_baseline_v2.json
FT_RES=results/m12_test_finetuned_${RUN}.json

if [[ -z "${FORCE:-}" && ( -e "$BASE_RES" || -e "$FT_RES" ) ]]; then
  echo "test results already exist ($BASE_RES / $FT_RES); not re-running test. Set FORCE=1 to override." >&2
  exit 1
fi
test -f "$ADAPTER/adapter_config.json" || { echo "adapter not found: $ADAPTER" >&2; exit 1; }

mkdir -p results data/claims/predictions
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "git commit: $(git rev-parse --short HEAD)   adapter: $ADAPTER"

echo "=== prompted baseline (v2) on test ==="
python scripts/baseline_claims.py --split test --prompt v2 \
  --out "$BASE_PRED" --raw-out "$RUNS/m12_test_baseline_v2_raw.jsonl"
python scripts/score_predictions.py claims --pred "$BASE_PRED" \
  --gold data/claims/splits/test.jsonl --out "$BASE_RES"

echo "=== fine-tuned adapter on test ==="
python scripts/predict_claims.py --adapter "$ADAPTER" --split test \
  --out "$FT_PRED" --raw-out "$RUNS/m12_test_finetuned_${RUN}_raw.jsonl"
python scripts/score_predictions.py claims --pred "$FT_PRED" \
  --gold data/claims/splits/test.jsonl --out "$FT_RES"

echo "=== error listings ==="
python scripts/claim_errors.py --pred "$BASE_PRED" --gold data/claims/splits/test.jsonl \
  --out results/m12_test_errors_baseline_v2.txt > /dev/null
python scripts/claim_errors.py --pred "$FT_PRED" --gold data/claims/splits/test.jsonl \
  --out "results/m12_test_errors_finetuned_${RUN}.txt" > /dev/null
echo "=== done ==="
