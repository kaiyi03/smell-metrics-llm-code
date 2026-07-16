---
title: Code Smell Evaluation Dashboard
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Code Smell Evaluation Dashboard

Paste Python code and see:

- the **code smells** it contains, each with the **line and rule** that flags it,
- its **structural measures** (complexity, maintainability, size, …) placed against a
  clean-code baseline (percentile),
- for a detected smell, where the code sits between clean and smelly medians,
- **similarity** to a reference solution (BLEU, CodeBLEU, AST edit distance, …).

Part of a University of Oxford research project on benchmarking poorly-structured
("smelly") LLM-generated Python code.

**Note:** the correctness check runs submitted code, so it is disabled on this hosted
demo. Run the dashboard locally for that — source: https://github.com/kaiyi03/Code-Smells
