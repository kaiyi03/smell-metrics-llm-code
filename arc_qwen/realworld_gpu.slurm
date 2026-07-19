#!/bin/bash
# Refresh the REAL-CODE detection strength on a GPU node, so realworld_results.csv
# includes the reference-free measures added this session -- cognitive complexity,
# comment density, API usage -- and perplexity (which needs the GPU).
# Submit:  sbatch arc_qwen/realworld_gpu.slurm
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --account=engs-applied-machine-learning
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=smell-real
#SBATCH --output=/home/some5943/realworld_gpu_%j.log

set -e
source ~/smell-env.sh
export HF_HOME=/data/engs-applied-machine-learning/some5943/hf
REPO=${REPO:-~/Code-Smells}

# Deps should persist from the scoring job, but reinstall cheaply just in case.
pip install -q numpy radon pylint ruff sacrebleu rouge-score nltk \
    cognitive_complexity zss tree-sitter tree-sitter-python \
    bert-score code-bert-score || true
pip install -q --no-deps codebleu || true

export PPL_MODEL=${PPL_MODEL:-Salesforce/codegen-350M-mono}
echo "node: $(hostname)"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"

cd "$REPO"
python eval_tool/run_realworld.py

echo
echo "DONE. Copy back: eval_tool/realworld_results.csv"
