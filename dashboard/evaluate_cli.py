"""Evaluate a snippet from the command line -- no server, no browser, no port.

Runs the SAME detectors and measures as the web dashboard and prints a plain-text
report. Use this whenever the web server is inconvenient or won't stay up.

  .venv/Scripts/python dashboard/evaluate_cli.py mycode.py
  .venv/Scripts/python dashboard/evaluate_cli.py mycode.py --ref clean.py --tests t.py --run
  type mycode.py | .venv/Scripts/python dashboard/evaluate_cli.py -
"""

import argparse
import bisect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the dashboard's own logic so the CLI and the web UI can never drift apart.
from app import BASELINES, PROFILE, SIM, SMELLY, STRUCT, detect_labeled, run_program  # noqa: E402


def _read(arg):
    if not arg:
        return ""
    if arg == "-":
        return sys.stdin.read()
    with open(arg, encoding="utf-8") as f:
        return f.read()


def _pctile(name, value):
    b = BASELINES.get(name)
    if not b or value is None:
        return ""
    p = bisect.bisect_right(b["dist"], value)
    if name in PROFILE:
        return f"p{p}"
    if b["worse"] == "up":
        tier = "high" if p >= 95 else "elevated" if p >= 75 else "typical"
    else:
        tier = "very-low" if p <= 5 else "low" if p <= 25 else "typical"
    return f"p{p} {tier}"


def _num(v, dp=2):
    return "-" if v is None else f"{v:.{dp}f}"


def main():
    ap = argparse.ArgumentParser(description="Evaluate a Python snippet (no server).")
    ap.add_argument("code", help="Python file to evaluate, or - for stdin")
    ap.add_argument("--ref", help="reference solution -> enables similarity measures")
    ap.add_argument("--tests", help="test block -> checks correctness (needs --run)")
    ap.add_argument("--run", action="store_true", help="actually execute the test block")
    args = ap.parse_args()

    code, ref, tests = _read(args.code), _read(args.ref), _read(args.tests)
    if not code.strip():
        sys.exit("no code given")

    name = args.code if args.code != "-" else "stdin"
    print(f"\n=== Evaluation: {name} ===\n")

    # --- detectors ---
    smells, problems = detect_labeled(code)
    print("SMELL DETECTORS")
    if problems:
        print("  [detector did NOT run] " + "; ".join(problems))
    elif smells:
        print("  [!] " + ", ".join(smells))
    else:
        print("  no tracked smells detected (the detectors are threshold-based; "
              "clean by their rules)")

    # --- structural, with clean-baseline percentile ---
    vals = {m.name: m.fn(code) for m in STRUCT}
    print("\nSTRUCTURAL MEASURES        value      vs clean")
    for m in STRUCT:
        print(f"  {m.name:22} {_num(vals[m.name]):>7}    {_pctile(m.name, vals[m.name])}")

    # --- clean -> you -> smelly for each detected smell ---
    for s in smells:
        rows = []
        for m in STRUCT:
            info = SMELLY.get((s, m.name))
            if info and info["d"] is not None and info["d"] >= 1.0:
                rows.append((info["d"], m.name, info["clean"], vals[m.name], info["smelly"]))
        rows.sort(reverse=True)
        if rows:
            print(f"\nWHERE YOU SIT ({s})     clean      you      smelly")
            for _, mn, clean, you, smelly in rows[:4]:
                print(f"  {mn:22} {_num(clean):>7}  {_num(you):>7}  {_num(smelly):>7}")

    # --- similarity (needs a reference) ---
    if ref.strip():
        print("\nSIMILARITY vs reference (100 = identical)")
        for m in SIM:
            print(f"  {m.name:22} {_num(m.fn(code, ref), 1):>7}")

    # --- correctness (needs tests + --run) ---
    if tests.strip():
        if args.run:
            print(f"\nCORRECTNESS: {run_program(code + chr(10) * 2 + tests).upper()}")
        else:
            print("\nCORRECTNESS: tests provided but not run (pass --run to execute them)")
    print()


if __name__ == "__main__":
    main()
