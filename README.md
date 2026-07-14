# Benchmarking and Metrics for Poorly Structured ("Smelly") LLM-Generated Code

This repository holds my contribution to a research project at the University of Oxford's Machine Learning Research Group. My part studies how much poorly structured — "smelly" — code large language models produce, and, more usefully, *which automated measures actually detect those smells*.

The work has two parts:

1. **A labelled benchmark** of clean vs. smelly Python code — 1,300 matched examples across 12 code smells.
2. **An evaluation tool** that scores code on three dimensions — structural quality, correctness, and generation cost — using a panel of independent measures.

## Live visualizations

The rendered reports are published with GitHub Pages — **[browse them here →](https://kaiyi03.github.io/Code-Smells/)**:

- [Smell guide](https://kaiyi03.github.io/Code-Smells/smell_injection/smell_guide.html) — the 12 smells, how each is injected, with real clean-vs-smelly examples
- [Detection-strength report](https://kaiyi03.github.io/Code-Smells/eval_tool/detection_report.html) — how strongly each measure separates smelly from clean code, on the injected pairs and on real code side by side
- [Qwen evaluation](https://kaiyi03.github.io/Code-Smells/arc_qwen/evaluation_report.html) — the evaluation tool applied to the model's generated code
- [Qwen generations](https://kaiyi03.github.io/Code-Smells/arc_qwen/generations_report.html) — the model's solutions next to the canonical answers
- [Sample browser](https://kaiyi03.github.io/Code-Smells/smell_injection/samples_report.html) — every clean/smelly pair, side by side

The same `.html` files live in the repo, but GitHub shows them as source; the links above open them rendered.

## Why

Metrics like BLEU, cyclomatic complexity, and CodeBLEU are routinely used to judge generated code, but it is rarely clear what each one actually captures. I build a controlled benchmark where I know exactly which smell is present, then test which measures detect it. The early result: **no single measure catches every smell, but the measures are complementary**, so a panel works where any one alone fails.

## Repository structure

```
smell_injection/       the benchmark: inject one smell into clean code, verify with a detector
  build_injected.py      generates the dataset (AST-based injection + detector confirmation)
  import_reused.py       imports and de-duplicates reused public code-smell datasets
  view_samples.py        renders a readable clean-vs-smelly HTML report
  samples.jsonl          the benchmark: 1,300 labelled clean/smelly pairs

eval_tool/             the evaluation tool: a panel of independent measures
  measures.py            the measures (structural + similarity families)
  run_panel.py           runs the panel over the benchmark, writes results + report
  run_realworld.py       runs the structural measures on real labelled code (second source)
  detection_report.py    merges injected + real detection strength into one report
  correctness.py         runs each function's tests in a sandbox (the correctness dimension)
  panel_results.csv      per (smell, measure) results

dashboard/             the interactive evaluation dashboard (local Flask app)
  app.py                 paste code (+ optional reference/tests) -> smells, measures, correctness

arc_qwen/              generating and scoring a model's code (Qwen2.5-Coder)
  generate.py            loads the model, generates solutions for the benchmark tasks
  gen.slurm              SLURM job script (Oxford ARC GPU cluster)
  view_generations.py    renders a readable side-by-side HTML report
  generations.jsonl      664 generated solutions (164 HumanEval + 500 MBPP)

docs/                  the research write-up
```

## The benchmark

Each example is a matched pair: the same function once clean and once with exactly one smell injected. I take a known-correct function (from MBPP and HumanEval, topped up with real GitHub functions from CodeSearchNet for the performance smells), confirm it is free of all target smells, mechanically inject one smell via an AST transform, and re-run the detection tool to confirm the smell landed and nothing else changed. The result is 1,300 labelled examples — 100 pairs for each of 12 smells, plus 100 clean-only examples.

The 12 smells: long method, long parameter list, deep nesting, too many boolean expressions, broad exception handling, unused import, mutable default argument, magic number (Pylint / Ruff); three performance anti-patterns (Ruff); and duplicate code (jscpd).

I treat the detectors' output as an **operational definition** of each smell, not absolute ground truth, and I never evaluate those detectors against their own labels — that would be circular.

## The evaluation tool

Given a piece of code, the tool computes a panel of independent measures:

- **Structural** (reference-free): lines of code, cyclomatic complexity, cognitive complexity, maintainability index, Halstead volume/difficulty/effort (from Radon), plus comment-density and function/API-usage profiles.
- **Similarity** (reference-based): BLEU, chrF, ROUGE-L, METEOR, CodeBLEU, and AST-skeleton similarity, against a correct reference.
- **Correctness** (execution): the function is run against its real test cases.

To compare measures that live on different scales, I compute a single **detection-strength score** — a standardised effect size for how cleanly each measure separates smelly code from its clean twin (0 = blind, ~1 = clear, 5 = capped, i.e. essentially perfect). Concretely this is **Cohen's d** — the gap between the clean and smelly means divided by the pooled standard deviation (how much the measure naturally varies), capped at 5. The same statistic is computed on real labelled code, so the injected and real detection strengths are directly comparable. `run_panel.py` writes the results CSV; `run_realworld.py` adds the real-code side; `detection_report.py` merges both into one colour-coded report. 

## Interactive dashboard

`dashboard/app.py` is a small local web app for evaluating a single snippet on demand. Paste code — optionally a reference solution and a test block — and it runs the same panel used on the benchmark: the smell detectors, the structural measures, similarity against the reference, and correctness against the tests. It imports the exact same measures as the offline tool, so the numbers match.

```bash
.venv/Scripts/python dashboard/app.py      # then open http://127.0.0.1:5000
```

## Model generation and scoring

`arc_qwen/` generates code with Qwen2.5-Coder-1.5B-Instruct for the benchmark tasks and scores it with the same panel — measuring how smelly, correct, and close-to-reference a real model's output is.

## Key findings so far

- No single measure catches all 12 smells, but every smell is caught by at least one.
- Structural and text measures are complementary: structural measures catch bulky or branch-heavy smells but miss "parameter-based" smells (long parameter list, mutable default); the text measures catch those.
- The code-aware measure (CodeBLEU) separates several smells more cleanly than plain BLEU.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
npm install jscpd                  # the duplicate-code detector is a Node tool
```

Then:

```bash
python smell_injection/build_injected.py   # build the benchmark
python eval_tool/run_panel.py              # score the measures, write the report
python eval_tool/correctness.py            # run the correctness check
```

## Data and sources

The clean reference code comes from MBPP, HumanEval, and CodeSearchNet (all permissively licensed). `smell_injection/reused.jsonl` is a normalised subset of existing public code-smell datasets (CodeSmellData, PySmell), each cited in the research document; `import_reused.py` shows how it is assembled from their original sources. Generated code comes from Qwen2.5-Coder.

## Status and next steps

Done: the benchmark, the evaluation tool (structural + similarity + correctness), model generation, and an interactive dashboard for scoring code on demand. Next: a runtime-performance measure (executed under controlled workloads, so timing is meaningful rather than noise) and the model-based measures (perplexity, CodeBERTScore, BERTScore) on the ARC GPU cluster.

## Acknowledgements

This work forms part of a larger research project at the University of Oxford. It used the University of Oxford's Advanced Research Computing (ARC) facility (http://dx.doi.org/10.5281/zenodo.22558) for the model-generation experiments. The full write-up is in `docs/`.
