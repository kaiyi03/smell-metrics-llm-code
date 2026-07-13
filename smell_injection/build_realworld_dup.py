"""
Mine real duplicate_code examples as CONCATENATED CLONE-PAIRS.

duplicate_code is duplication WITHIN a code unit, and the injector builds it that
way -- it appends a renamed copy of a function, so one snippet holds both copies.
To compare like with like, we build the real examples the same way: run jscpd
across a batch of real methods to find genuine clone pairs (A near-identical to B),
then concatenate each pair into one snippet. That gives a real "unit that contains
a duplicate", matching the injected shape.

The point of doing this is honesty, not a detection number: a duplicated block has
the same structure as the original, so any structural "signal" for duplicate_code
-- injected or real -- is only the SIZE of the second copy, not the duplication.
The real concatenated pairs should reproduce the injected size-driven signal,
confirming that a clone detector (jscpd), not the structural measures, is what
actually finds duplication.

Appends duplicate_code rows to realworld_smelly.jsonl (idempotent: strips prior
duplicate_code rows first). Run AFTER the other mining scripts.

Run:  python smell_injection/build_realworld_dup.py [--sample 3500 --target 250]
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
    ap.add_argument("--sample", type=int, default=3500, help="methods to scan together with jscpd")
    ap.add_argument("--target", type=int, default=250, help="concatenated clone-pairs to keep")
    args = ap.parse_args()
    if not JSCPD:
        raise SystemExit("jscpd not found.")

    raw = json.load(open(RAW, encoding="utf-8"))
    order = list(range(len(raw)))
    random.Random(2).shuffle(order)
    tmp = tempfile.mkdtemp()
    code_of = {}
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
            code_of[fn] = code
            n += 1
        print(f"scanning {n} real methods with jscpd for clone pairs ...")

        out = os.path.join(tmp, "report")
        cmd = [JSCPD, tmp, "--min-lines", str(DUP_MIN_LINES), "--min-tokens", str(DUP_MIN_TOKENS),
               "--reporters", "json", "--output", out, "--silent"]
        if os.name == "nt":
            cmd = ["cmd", "/c", *cmd]
        subprocess.run(cmd, capture_output=True, text=True)

        pairs, seen = [], set()
        report = os.path.join(out, "jscpd-report.json")
        if os.path.exists(report):
            for dup in json.load(open(report, encoding="utf-8")).get("duplicates", []):
                a = os.path.basename((dup.get("firstFile") or {}).get("name", ""))
                b = os.path.basename((dup.get("secondFile") or {}).get("name", ""))
                key = frozenset((a, b))
                if a in code_of and b in code_of and a != b and key not in seen:
                    seen.add(key)
                    pairs.append((a, b))
                    if len(pairs) >= args.target:
                        break
        print(f"jscpd found {len(pairs)} distinct clone pairs")

        recs = []
        for a, b in pairs:
            # concatenate the two clones into one snippet -- matches the injector's
            # "function + its copy" shape.
            snippet = code_of[a].rstrip() + "\n\n\n" + code_of[b].rstrip() + "\n"
            recs.append({"id": f"cs2dup_{a[1:-3]}_{b[1:-3]}", "source": "codesmelldata2_mined",
                         "smell": "duplicate_code", "label": "yes", "code": snippet,
                         "label_origin": "jscpd", "meta": {"kind": "concatenated clone-pair"}})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    existing = []
    if os.path.exists(SMELLY):
        existing = [json.loads(l) for l in open(SMELLY, encoding="utf-8") if l.strip()
                    and json.loads(l).get("smell") != "duplicate_code"]
    with open(SMELLY, "w", encoding="utf-8") as f:
        for r in existing + recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} concatenated duplicate_code snippets "
          f"({len(existing) + len(recs)} total mined rows)")


if __name__ == "__main__":
    main()
