"""
Mine real duplicate_code examples -- the one smell single-method mining misses.

duplicate_code is an ACROSS-unit clone, so a single method almost never contains
one. But real methods ARE clones of *other* methods (copy-pasted helpers, near
-identical handlers). We dump a large batch of real methods from the CodeSmellData
2.0 raw corpus into one folder, run jscpd across the whole folder, and keep the
methods jscpd flags as clones of another method -> real duplicate_code positives.

These get appended to realworld_smelly.jsonl so run_realworld.py picks them up.
(Run this AFTER build_realworld_smelly.py; it is idempotent -- it strips any prior
duplicate_code rows before appending.)

Expected finding: a cloned method, viewed on its own, is structurally ordinary, so
the structural measures should show ~0 separation -- confirming duplicate_code is a
clone-detector smell that structural measures cannot see on a single unit, and that
the injected structural signal (which doubles the snippet) is an artefact.

Run:  python smell_injection/build_realworld_dup.py [--sample 3000]
"""

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile


def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "Scripts", "python.exe"))
    if not os.path.exists(venv_py):
        venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "bin", "python"))
    if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != venv_py.lower():
        print(f"[setup] switching to project venv:\n        {venv_py}\n")
        raise SystemExit(subprocess.run([venv_py, os.path.abspath(__file__), *sys.argv[1:]]).returncode)


_bootstrap()

import random

from build_injected import JSCPD, DUP_MIN_LINES, DUP_MIN_TOKENS

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, os.pardir, "_datasets_eval", "CodeSmellExt",
                   "dataset", "extracted", "CodeSmellData_2.0.json")
SMELLY = os.path.join(HERE, "realworld_smelly.jsonl")
MIN_LINES = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=3000, help="methods to scan together with jscpd")
    args = ap.parse_args()
    if not JSCPD:
        raise SystemExit("jscpd not found (npm install -g jscpd, or local node_modules).")

    raw = json.load(open(RAW, encoding="utf-8"))
    order = list(range(len(raw)))
    random.Random(2).shuffle(order)
    tmp = tempfile.mkdtemp()
    fname_to_code = {}
    try:
        n = 0
        for idx in order:
            if n >= args.sample:
                break
            code = (raw[idx].get("code") or "").strip()
            if not code or len(code.splitlines()) < MIN_LINES:
                continue
            try:
                ast.parse(code)
            except (SyntaxError, ValueError):
                continue
            fn = f"m{idx}.py"
            with open(os.path.join(tmp, fn), "w", encoding="utf-8") as f:
                f.write(code)
            fname_to_code[fn] = (idx, code)
            n += 1
        print(f"scanning {n} real methods together with jscpd (min {DUP_MIN_LINES} lines) ...")

        out = os.path.join(tmp, "report")
        cmd = [JSCPD, tmp, "--min-lines", str(DUP_MIN_LINES), "--min-tokens", str(DUP_MIN_TOKENS),
               "--reporters", "json", "--output", out, "--silent"]
        if os.name == "nt":
            cmd = ["cmd", "/c", *cmd]
        subprocess.run(cmd, capture_output=True, text=True)
        report = os.path.join(out, "jscpd-report.json")
        cloned = set()
        if os.path.exists(report):
            data = json.load(open(report, encoding="utf-8"))
            for dup in data.get("duplicates", []):
                for side in ("firstFile", "secondFile"):
                    name = os.path.basename((dup.get(side) or {}).get("name", ""))
                    if name in fname_to_code:
                        cloned.add(name)
        print(f"jscpd flagged {len(cloned)} methods as clones of another method")

        recs = []
        for fn in cloned:
            idx, code = fname_to_code[fn]
            recs.append({"id": f"cs2dup_{idx}", "source": "codesmelldata2_mined",
                         "smell": "duplicate_code", "label": "yes", "code": code,
                         "label_origin": "jscpd", "meta": {"repo": raw[idx].get("repo")}})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # append to realworld_smelly.jsonl, idempotently (strip prior duplicate_code rows)
    existing = []
    if os.path.exists(SMELLY):
        existing = [json.loads(l) for l in open(SMELLY, encoding="utf-8") if l.strip()
                    and json.loads(l).get("smell") != "duplicate_code"]
    with open(SMELLY, "w", encoding="utf-8") as f:
        for r in existing + recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} duplicate_code rows into {os.path.basename(SMELLY)} "
          f"({len(existing) + len(recs)} total mined rows)")


if __name__ == "__main__":
    main()
