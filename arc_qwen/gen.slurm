#!/bin/bash
# SLURM job for Qwen generation on ARC (htc GPU cluster).
# Submit a test:  sbatch gen.slurm --limit 5 --out ~/generations_test.jsonl
# Submit full:    sbatch gen.slurm --out ~/generations.jsonl
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=engs-applied-machine-learning
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=qwen-gen
#SBATCH --output=/home/some5943/gen_%j.log

source ~/smell-env.sh
export HF_HOME=/data/engs-applied-machine-learning/some5943/hf
echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd ~
python generate.py "$@"
