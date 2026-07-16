#!/bin/bash
# SLURM job for the model-based (GPU) measures on ARC (htc GPU cluster).
#
# It runs the SAME panel scripts as on the laptop, but on a GPU node, so the three
# model measures (perplexity, CodeBERTScore, BERTScore) switch on and get folded into:
#   * benchmark detection strength   run_panel.py         -> eval_tool/panel_results.csv
#   * Qwen deployment scoring         evaluate_generations.py -> arc_qwen/evaluation_summary.csv
#
# Prereq: clone/pull this repo on ARC (default location ~/Code-Smells). Then:
#   sbatch arc_qwen/score_gpu.slurm
#
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=engs-applied-machine-learning
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=smell-gpu
#SBATCH --output=/home/some5943/score_gpu_%j.log

set -e
source ~/smell-env.sh
export HF_HOME=/data/engs-applied-machine-learning/some5943/hf

# Where the repo is checked out on ARC. Override with: REPO=/path sbatch ...
REPO=${REPO:-~/Code-Smells}

# The ARC env has torch/transformers (for generation) but not the measure deps.
# Install them all (fast -- most are pure-Python wheels). codebleu goes in with
# --no-deps so its old tree-sitter pin doesn't fight tree-sitter-python.
pip install -q numpy radon pylint ruff sacrebleu rouge-score nltk \
    cognitive_complexity zss tree-sitter tree-sitter-python \
    bert-score code-bert-score || true
pip install -q --no-deps codebleu || true
python -m nltk.downloader -q wordnet omw-1.4 || true

# Code language model for perplexity (small + code-appropriate). Change to a bigger
# code model, e.g. the Qwen coder, for a stronger signal.
export PPL_MODEL=${PPL_MODEL:-Salesforce/codegen-350M-mono}

echo "node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

cd "$REPO"
echo "=== [1/2] benchmark detection strength (now incl. model measures) ==="
python eval_tool/run_panel.py

echo "=== [2/2] Qwen deployment scoring (now incl. model measures) ==="
python arc_qwen/evaluate_generations.py

echo
echo "DONE. Copy these back to your laptop and commit:"
echo "  eval_tool/panel_results.csv          (detection strength, now with the 3 model measures)"
echo "  arc_qwen/evaluation_summary.csv      (Qwen scoring, now with the 3 model measures)"
