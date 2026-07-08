"""
Generate code for the benchmark tasks with Qwen (runs on an ARC GPU node).

Loads Qwen2.5-Coder-1.5B-Instruct, prompts it to solve each MBPP / HumanEval
task, and records the generated code plus generation cost. The output
(generations.jsonl) is copied back to the laptop and scored with the eval panel.

Runs on a compute node (real memory + internet); the HF cache lives on the
shared /data area so the model is downloaded once and reused.

Usage (inside a SLURM GPU job -- see gen.slurm):
    python generate.py --limit 5 --out ~/generations_test.jsonl   # quick test
    python generate.py --out ~/generations.jsonl                  # full run
"""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")
MAX_NEW = 512
BATCH = 16


def load_tasks(limit=None):
    """Return the benchmark tasks as {task_id, source, instruction, canonical}."""
    tasks = []
    for ex in load_dataset("openai/openai_humaneval", split="test"):
        tasks.append({
            "task_id": ex["task_id"].replace("/", "_"),
            "source": "humaneval",
            "instruction": ("Complete the following Python function. Return only the "
                            "complete function in a single ```python code block.\n\n"
                            + ex["prompt"]),
            "canonical": ex["prompt"] + ex["canonical_solution"],
        })
    for ex in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        tests = "\n".join(ex["test_list"])
        tasks.append({
            "task_id": f"mbpp_{ex['task_id']}",
            "source": "mbpp",
            "instruction": ("Write a Python function for the task below. Return only the "
                            "code in a single ```python code block.\n\n"
                            f"Task: {ex['text']}\n\nIt must pass these tests:\n{tests}"),
            "canonical": ex["code"],
        })
    if limit:                                   # small, balanced slice for a quick test
        he = [t for t in tasks if t["source"] == "humaneval"][:limit]
        mb = [t for t in tasks if t["source"] == "mbpp"][:limit]
        return he + mb
    return tasks


def extract_code(text):
    """Pull the code out of the reply: prefer a fenced ```python block."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="tasks per source (test runs)")
    ap.add_argument("--out", default=os.path.expanduser("~/generations.jsonl"))
    args = ap.parse_args()

    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"                    # decoder-only batching needs left pad
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
    model.eval()

    tasks = load_tasks(args.limit)
    print(f"{len(tasks)} tasks; generating (batch {BATCH}, max_new {MAX_NEW}, greedy) ...",
          flush=True)

    n_done, n_tokens, t0 = 0, 0, time.time()
    with open(args.out, "w", encoding="utf-8") as fout:    # write incrementally
        for i in range(0, len(tasks), BATCH):
            batch = tasks[i:i + BATCH]
            prompts = [tok.apply_chat_template([{"role": "user", "content": t["instruction"]}],
                                               tokenize=False, add_generation_prompt=True)
                       for t in batch]
            enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
            n_in = enc["input_ids"].shape[1]
            st = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            dt = time.time() - st
            gen = out[:, n_in:]                              # strip the prompt tokens
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for t, g, ids in zip(batch, texts, gen):
                n_out = int((ids != tok.pad_token_id).sum())
                n_tokens += n_out
                fout.write(json.dumps({
                    "task_id": t["task_id"], "source": t["source"],
                    "generated_code": extract_code(g),
                    "raw_output": g,
                    "canonical_code": t["canonical"],
                    "n_prompt_tokens": int(n_in),
                    "n_output_tokens": n_out,
                    "batch_seconds": round(dt, 2),
                }) + "\n")
                fout.flush()
            n_done += len(batch)
            print(f"  {n_done}/{len(tasks)}  ({dt:.1f}s/batch)", flush=True)

    total = time.time() - t0
    print(f"wrote {n_done} generations to {args.out} in {total:.0f}s "
          f"({n_tokens / total:.0f} output tok/s overall)", flush=True)


if __name__ == "__main__":
    main()
