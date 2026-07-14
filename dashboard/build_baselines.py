"""Precompute the clean-code baseline for each structural measure.

The dashboard needs a reference to interpret a single reading -- "is a Halstead
difficulty of 2.65 high?" only means something against what CLEAN code scores. We
take that reference from the benchmark's real-clean pool (realworld_clean.jsonl,
1000 real functions our detectors found no smell in), run every structural measure
over it, and store a 101-point percentile curve per measure. The dashboard loads
this once and reports where a reading falls (its percentile in clean code) instead
of recomputing 1000 functions on every launch.

The SMELLY side of the baseline is smell-specific and already lives in the
evaluation tables: eval_tool/panel_results.csv (injected) and
eval_tool/realworld_results.csv (real) both carry clean_median and smelly_median
per (smell, measure).

Run:  .venv/Scripts/python dashboard/build_baselines.py
"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
sys.path.insert(0, os.path.join(ROOT, "eval_tool"))

from measures import PANEL   # noqa: E402

CLEAN = os.path.join(ROOT, "smell_injection", "realworld_clean.jsonl")
OUT = os.path.join(HERE, "baselines.json")
STRUCT = [m for m in PANEL if not m.needs_ref]


def curve(sorted_vals):
    """101 points: the value at each integer percentile p0..p100."""
    n = len(sorted_vals)
    return [sorted_vals[min(n - 1, max(0, round(p / 100 * (n - 1))))] for p in range(101)]


def main():
    rows = [json.loads(line) for line in open(CLEAN, encoding="utf-8")]
    code_key = next(k for k in rows[0] if isinstance(rows[0][k], str) and "def " in rows[0][k])

    base = {}
    for m in STRUCT:
        vals = sorted(v for v in (m.fn(r[code_key]) for r in rows) if v is not None)
        if not vals:
            continue
        base[m.name] = {
            "n": len(vals),
            "worse": m.worse,                       # 'up' = higher is worse, 'down' = lower is worse
            "median": round(statistics.median(vals), 3),
            "p95": round(vals[min(len(vals) - 1, round(0.95 * (len(vals) - 1)))], 3),
            "dist": [round(v, 4) for v in curve(vals)],
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"source": "realworld_clean.jsonl", "n_functions": len(rows), "clean": base}, f)
    print(f"wrote {OUT}  |  {len(base)} measures over {len(rows)} clean functions")


if __name__ == "__main__":
    main()
