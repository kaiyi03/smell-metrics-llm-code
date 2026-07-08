"""
Correctness by execution.

The structural and similarity measures never actually RUN the code. This does:
it executes each function against its task's real test cases and records
pass / fail / timeout.

Two jobs:
  * Now (validation): run the tests on the clean and the smelly half of every
    injected pair. The injectors are meant to be behaviour-preserving (they add
    defaulted params, dead statements, once-round loops, etc.), so a smelly
    function should still pass whatever its clean twin passes. This confirms the
    dataset is sound and flags any smell that secretly changes behaviour.
  * Later (deployment): the same run_tests() scores the Qwen model's output.

Tests come from MBPP and HumanEval. CodeSearchNet samples have no tests, so they
are reported separately as "no test available", not counted as failures.

Safety: each snippet runs in a separate Python process with a wall-clock
timeout and no arguments. That is enough for our own dataset and for a local
baseline model. Running genuinely untrusted third-party code would want stronger
isolation (a container); noted for the write-up.

Run:  python eval_tool/correctness.py        (auto-switches to the project venv)
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "Scripts", "python.exe"))
    if not os.path.exists(venv_py):  # linux / ARC layout
        venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "bin", "python"))
    if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != venv_py.lower():
        print(f"[setup] switching to project venv:\n        {venv_py}\n")
        raise SystemExit(subprocess.run([venv_py, os.path.abspath(__file__), *sys.argv[1:]]).returncode)


_bootstrap()

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, os.pardir, "smell_injection", "samples.jsonl")
OUT_CSV = os.path.join(HERE, "correctness_results.csv")
TIMEOUT = 10        # seconds per snippet
WORKERS = 8         # parallel subprocesses


def load_pairs():
    rows = [json.loads(line) for line in open(SAMPLES, encoding="utf-8")]
    return [r for r in rows if r.get("smell") and r["smell"] != "clean"]


def load_tests():
    """task-id -> test spec, matching the ids used as source_task in samples.jsonl."""
    from datasets import load_dataset
    tests = {}
    for split in ("train", "test", "validation"):
        for ex in load_dataset("google-research-datasets/mbpp", "full", split=split):
            tests[f"mbpp_{ex['task_id']}"] = {
                "kind": "mbpp",
                "setup": ex.get("test_setup_code") or "",
                "asserts": list(ex["test_list"]),
            }
    for ex in load_dataset("openai/openai_humaneval", split="test"):
        tests[ex["task_id"].replace("/", "_")] = {
            "kind": "humaneval",
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        }
    return tests


def build_program(code, spec):
    """Assemble a self-contained script that raises if the code is wrong."""
    if spec["kind"] == "mbpp":
        parts = [code, spec["setup"], *spec["asserts"]]
        return "\n".join(p for p in parts if p)
    return f"{code}\n{spec['test']}\ncheck({spec['entry_point']})\n"


def run_program(program):
    """pass = tests ran with no error; fail = assertion/exception; timeout = hung."""
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(program)
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=TIMEOUT)
        return "pass" if proc.returncode == 0 else "fail"
    except subprocess.TimeoutExpired:
        return "timeout"
    except Exception:
        return "fail"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_map(jobs):
    """jobs: dict key -> program string. Returns key -> result, run in parallel."""
    out = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_program, prog): key for key, prog in jobs.items()}
        for fut in futures:
            out[futures[fut]] = fut.result()
    return out


def main():
    rows = load_pairs()
    print("loading MBPP + HumanEval tests ...")
    tests = load_tests()

    testable = [r for r in rows if r["source_task"] in tests]
    no_test = len(rows) - len(testable)
    print(f"{len(testable)} of {len(rows)} injected pairs have tests "
          f"({no_test} have none -- CodeSearchNet-sourced, skipped)\n")

    # Clean code is identical across a task's smells, so test each clean once.
    clean_jobs = {r["source_task"]: build_program(r["clean_code"], tests[r["source_task"]])
                  for r in testable}
    smelly_jobs = {r["id"]: build_program(r["smelly_code"], tests[r["source_task"]])
                   for r in testable}
    print(f"running {len(clean_jobs)} clean + {len(smelly_jobs)} smelly snippets "
          f"in a sandbox ({WORKERS} at a time) ...")
    clean_res = _run_map(clean_jobs)
    smelly_res = _run_map(smelly_jobs)

    # Aggregate per smell.
    by_smell = defaultdict(list)
    for r in testable:
        by_smell[r["smell"]].append(r)

    print(f"\n{'smell':<20} {'n':>4} {'clean pass':>11} {'smelly pass':>12} "
          f"{'behaviour kept':>15}")
    print("-" * 66)
    detail = []
    for smell in sorted(by_smell):
        group = by_smell[smell]
        n = len(group)
        clean_ok = sum(clean_res[r["source_task"]] == "pass" for r in group)
        smelly_ok = sum(smelly_res[r["id"]] == "pass" for r in group)
        # behaviour kept = of the pairs whose clean twin passes, how many smelly also pass
        base = [r for r in group if clean_res[r["source_task"]] == "pass"]
        kept = sum(smelly_res[r["id"]] == "pass" for r in base)
        kept_pct = (kept / len(base) * 100) if base else float("nan")
        print(f"{smell:<20} {n:>4} {clean_ok / n * 100:>10.0f}% "
              f"{smelly_ok / n * 100:>11.0f}% "
              f"{kept_pct:>14.0f}%")
        detail.append((smell, n, clean_ok, smelly_ok, len(base), kept))

    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("smell,n_tested,clean_pass,smelly_pass,clean_pass_base,behaviour_kept\n")
        for smell, n, c, s, base, kept in detail:
            f.write(f"{smell},{n},{c},{s},{base},{kept}\n")
    print(f"\nwrote {OUT_CSV}")

    # Flag anything suspicious for the write-up.
    broken = [smell for smell, n, c, s, base, kept in detail if base and kept < base]
    if broken:
        print(f"\nNOTE: these smells changed behaviour in some pairs (worth a look): "
              f"{', '.join(broken)}")
    else:
        print("\nAll injectors preserved behaviour wherever the clean twin passed.")


if __name__ == "__main__":
    main()
