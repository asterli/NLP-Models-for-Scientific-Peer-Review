#!/bin/bash
#SBATCH --job-name=m12_check
#SBATCH --account=xiangren_1987
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --constraint="a40|l40s|a100"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:20:00
#SBATCH --output=/project2/xiangren_1987/group_40/runs/logs/m12_check_%j.out
#
# M12: check that the callable extractor (src/claims/extractor.py) reproduces the committed
# fine-tuned DEV predictions. Does not touch test.
#   sbatch scripts/slurm/m12_extractor_check.sbatch

set -euo pipefail
module load conda && eval "$(conda shell.bash hook)"
conda activate /project2/xiangren_1987/group_40/envs/nlp
export HF_HOME=/project2/xiangren_1987/group_40/hf_cache
cd /project2/xiangren_1987/group_40/NLP-Models-for-Scientific-Peer-Review

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "git commit: $(git rev-parse --short HEAD)"
python scripts/check_extractor.py --adapter /project2/xiangren_1987/group_40/runs/m11/r16_lr2e-4_s544/best
