"""
Boost the real perf-smell examples with a fast ruff-only sweep of the whole corpus.

The perf smells (inefficient_loop PERF401, inefficient_copy PERF402, perf_try_in_loop
PERF203) are ruff-only, and build_realworld_smelly.py ran the full (slow) pylint+ruff
detector on a small sample, so it found very few -- inefficient_copy only 2, which is
too small to conclude anything. Ruff alone is fast, so this sweeps the ENTIRE
CodeSmellData 2.0 raw corpus with just the three PERF rules and keeps every hit,
maximising the (inherently rare) yield.

It updates the perf-smell rows in realworld_smelly.jsonl in place, leaving the
magic_number and duplicate_code rows (mined by the other scripts) untouched.

Run:  python smell_injection/build_realworld_perf.py
"""

import ast
import json
import os
import subprocess
import sys


def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "Scripts", "python.exe"))
    if not os.path.exists(venv_py):
        venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "bin", "python"))
    if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != venv_py.lower():
        print(f"[setup] switching to project venv:\n        {venv_py}\n")
        raise SystemExit(subprocess.run([venv_py, os.path.abspath(__file__), *sys.argv[1:]]).returncode)


_bootstrap()

import shutil
import tempfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, os.pardir, "_datasets_eval", "CodeSmellExt",
                   "dataset", "extracted", "CodeSmellData_2.0.json")
SMELLY = os.path.join(HERE, "realworld_smelly.jsonl")
CODE_TO_SMELL = {"PERF203": "perf_try_in_loop", "PERF401": "inefficient_loop",
                 "PERF402": "inefficient_copy"}
CAP = {"perf_try_in_loop": 300, "inefficient_loop": 300, "inefficient_copy": 10 ** 9}
MIN_LINES = 2
CHUNK = 4000


def ruff_hits(folder):
    proc = subprocess.run([sys.executable, "-m", "ruff", "check", folder, "--isolated",
                           "--select", ",".join(CODE_TO_SMELL), "--output-format", "json"],
                          capture_output=True, text=True)
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return []


def main():
    raw = json.load(open(RAW, encoding="utf-8"))
    print(f"ruff-sweeping all {len(raw)} raw methods for {sorted(CODE_TO_SMELL.values())} ...")

    kept, seen, count, i = [], set(), Counter(), 0
    while i < len(raw) and any(count[s] < CAP[s] for s in CAP):
        tmp = tempfile.mkdtemp()
        fmap = {}
        try:
            while len(fmap) < CHUNK and i < len(raw):
                code = (raw[i].get("code") or "").strip()
                idx = i
                i += 1
                if not code or len(code.splitlines()) < MIN_LINES:
                    continue
                try:
                    ast.parse(code)
                except (SyntaxError, ValueError):
                    continue
                fn = f"m{idx}.py"
                with open(os.path.join(tmp, fn), "w", encoding="utf-8") as f:
                    f.write(code)
                fmap[fn] = (idx, code)
            if not fmap:
                continue
            for m in ruff_hits(tmp):
                fn = os.path.basename(m.get("filename") or "")
                smell = CODE_TO_SMELL.get(m.get("code"))
                if fn in fmap and smell and count[smell] < CAP[smell] and (fn, smell) not in seen:
                    idx, code = fmap[fn]
                    kept.append({"id": f"cs2mined_{idx}_{smell}", "source": "codesmelldata2_mined",
                                 "smell": smell, "label": "yes", "code": code,
                                 "label_origin": "ruff", "meta": {"repo": raw[idx].get("repo")}})
                    seen.add((fn, smell))
                    count[smell] += 1
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        if i % (CHUNK * 5) < CHUNK:
            print(f"  scanned {i}: " + ", ".join(f"{s}={count[s]}" for s in sorted(CODE_TO_SMELL.values())),
                  flush=True)

    # merge: keep everything except old perf rows, add the fresh perf rows
    existing = []
    if os.path.exists(SMELLY):
        for l in open(SMELLY, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r.get("smell") not in CODE_TO_SMELL.values():
                    existing.append(r)
    with open(SMELLY, "w", encoding="utf-8") as f:
        for r in existing + kept:
            f.write(json.dumps(r) + "\n")
    print(f"\nkept perf rows: {dict(count)}  (scanned {i})")
    print(f"realworld_smelly.jsonl now {len(existing) + len(kept)} rows")


if __name__ == "__main__":
    main()
