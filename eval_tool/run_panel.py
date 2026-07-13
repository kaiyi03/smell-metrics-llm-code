"""
Run the measure panel over the injected dataset and report how well each
INDEPENDENT measure separates clean code from its smelly twin.

This is the validation harness. samples.jsonl gives matched pairs (same
function -- one clean, one with a single injected smell), so for every
(smell, measure) we can check what the smell did to the measure.

Two families are read differently:

  structural (reference-free)  -- clean and smelly each get a real score.
      hit_rate = fraction of pairs where the smell moved the measure the
      "worse" way. ~1.0 = reliably notices this smell; ~0.5 = blind to it.

  similarity (reference-based) -- scored against the clean reference, so
      clean-vs-clean is 100 by construction. We report the median score the
      SMELLY code keeps (100 = smell changed nothing the metric sees; lower =
      the smell perturbed the code more).

Outputs three things: the two tables to the console, panel_results.csv (full
numbers), and panel_report.html (colour-coded, the readable view).

Run:  python eval_tool/run_panel.py        (auto-switches to the project venv)
"""

import json
import os
import subprocess
import sys
from collections import defaultdict


def _bootstrap():
    """Re-launch under the project .venv if we're not already on it, so the
    measure libraries are importable no matter how the script was started."""
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "Scripts", "python.exe"))
    if not os.path.exists(venv_py):  # linux / ARC layout
        venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "bin", "python"))
    if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != venv_py.lower():
        print(f"[setup] switching to project venv:\n        {venv_py}\n")
        raise SystemExit(subprocess.run([venv_py, os.path.abspath(__file__), *sys.argv[1:]]).returncode)


_bootstrap()

import numpy as np              # noqa: E402  (import after bootstrap)
from measures import PANEL      # noqa: E402
import report                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, os.pardir, "smell_injection", "samples.jsonl")
OUT_CSV = os.path.join(HERE, "panel_results.csv")
OUT_HTML = os.path.join(HERE, "panel_report.html")
CORR_CSV = os.path.join(HERE, "correctness_results.csv")

PERFECT = 100.0   # a similarity measure scores identical code at 100 by construction


def load_pairs():
    rows = [json.loads(line) for line in open(SAMPLES, encoding="utf-8")]
    # injected pairs only; the 'clean' negatives have no smelly twin to contrast
    return [r for r in rows if r.get("smell") and r["smell"] != "clean"]


def load_correctness():
    """Read correctness_results.csv (written by correctness.py) if it exists, so the
    report can show all three dimensions together. Returns None if it's not there."""
    if not os.path.exists(CORR_CSV):
        return None
    import csv
    out = {}
    for row in csv.DictReader(open(CORR_CSV, encoding="utf-8")):
        out[row["smell"]] = {k: int(v) for k, v in row.items() if k != "smell"}
    return out


def _median(vals):
    keep = [v for v in vals if v is not None]
    return float(np.median(keep)) if keep else float("nan")


D_CAP = 5.0   # display ceiling for Cohen's d (a perfectly consistent shift is "infinite")


def paired_stats(clean_vals, smelly_vals, worse):
    """Summarise what the smell did to one measure across the pairs.

    Returns n, hit_rate, delta_median, delta_mean, cohen_d. hit_rate is the paired
    direction test -- the fraction of pairs where the smell moved the measure the
    worse way (the scope's "rank the clean version above its smelly counterpart in
    matched pairs"). cohen_d is the UNPAIRED effect size: the shift in means
    standardised by how much the measure naturally VARIES (pooled std), oriented so
    a POSITIVE value means 'moved the worse way', capped at +/-D_CAP. ~0 = no
    signal, 0.8 = large, >2 = very strong separation.

    Why not the paired form (mean change / spread OF THE CHANGE)? Because an
    injector that adds a CONSTANT shift -- dead code and magic number each add one
    fixed line -- drives the within-pair spread to ~0, which saturated the paired d
    to the cap and reported a trivial one-line change as a perfect detector.
    Standardising by natural variation does not saturate, and it is the same
    statistic the real-world layer uses, so injected and real detection strengths
    become directly comparable."""
    cs, ss, hits = [], [], 0
    for c, s in zip(clean_vals, smelly_vals):
        if c is None or s is None:
            continue
        cs.append(c)
        ss.append(s)
        if (worse == "up" and s - c > 0) or (worse == "down" and s - c < 0):
            hits += 1
    n = len(cs)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    ca, sa = np.asarray(cs, dtype=float), np.asarray(ss, dtype=float)
    deltas = sa - ca
    mean = float(deltas.mean())
    pooled = float(np.sqrt((ca.std(ddof=1) ** 2 + sa.std(ddof=1) ** 2) / 2)) if n > 1 else 0.0
    if pooled == 0:
        dz = 0.0 if mean == 0 else float(np.sign(mean)) * D_CAP
    else:
        dz = mean / pooled
    dz = dz if worse == "up" else -dz                       # positive = worse
    dz = max(-D_CAP, min(D_CAP, dz))
    return n, hits / n, float(np.median(deltas)), mean, dz


def main():
    rows = load_pairs()
    smells = sorted({r["smell"] for r in rows})
    ref_free = [m for m in PANEL if not m.needs_ref]
    ref_based = [m for m in PANEL if m.needs_ref]
    print(f"loaded {len(rows)} injected pairs across {len(smells)} smells")
    print(f"panel: {len(ref_free)} structural + {len(ref_based)} similarity measures\n")

    # Collect, per (smell, measure), a clean list and a smelly list, paired by index.
    clean = defaultdict(lambda: defaultdict(list))
    smelly = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ref_free:
            clean[r["smell"]][m.name].append(m.fn(r["clean_code"]))
            smelly[r["smell"]][m.name].append(m.fn(r["smelly_code"]))
        for m in ref_based:
            clean[r["smell"]][m.name].append(PERFECT)                 # clean-vs-clean
            smelly[r["smell"]][m.name].append(m.fn(r["smelly_code"], r["clean_code"]))

    # One stats dict per (smell, measure) -- feeds the console, the CSV and the HTML.
    cells = {}
    for s in smells:
        for m in PANEL:
            n, hr, dmed, dmean, dz = paired_stats(clean[s][m.name], smelly[s][m.name], m.worse)
            cells[(s, m.name)] = {
                "smell": s, "measure": m.name, "family": m.family, "n": n,
                "hit_rate": hr,
                "clean_median": _median(clean[s][m.name]),
                "smelly_median": _median(smelly[s][m.name]),
                "delta_median": dmed, "delta_mean": dmean, "cohen_d": dz,
            }

    # ---- console ------------------------------------------------------------
    print("STRUCTURAL  -- hit rate: % of pairs where the smell moved the measure the worse way")
    _print_matrix(smells, ref_free, lambda s, m: _fmt_pct(cells[(s, m.name)]["hit_rate"]))
    print("\nSIMILARITY  -- similarity score of the SMELLY code vs its clean twin "
          "(100 = identical; lower = the smell changed more)")
    _print_matrix(smells, ref_based, lambda s, m: _fmt_num(cells[(s, m.name)]["smelly_median"]))

    # ---- CSV ----------------------------------------------------------------
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("smell,measure,family,n_pairs,hit_rate,clean_median,smelly_median,"
                "delta_median,delta_mean,cohen_d\n")
        for s in smells:
            for m in PANEL:
                c = cells[(s, m.name)]
                f.write(f"{s},{m.name},{c['family']},{c['n']},{c['hit_rate']:.4f},"
                        f"{c['clean_median']:.4f},{c['smelly_median']:.4f},"
                        f"{c['delta_median']:.4f},{c['delta_mean']:.4f},{c['cohen_d']:.4f}\n")
    print(f"\nwrote {OUT_CSV}")

    # ---- HTML ---------------------------------------------------------------
    report.write_html(cells, smells, [m.name for m in ref_free],
                      [m.name for m in ref_based], OUT_HTML, len(rows),
                      correctness=load_correctness())
    print(f"wrote {OUT_HTML}  <- open this one")


def _print_matrix(smells, measures, cell):
    w = max(len(s) for s in smells)
    names = [m.name for m in measures]
    header = "smell".ljust(w) + "   " + " ".join(n[:11].rjust(11) for n in names)
    print(header)
    print("-" * len(header))
    for s in smells:
        print(s.ljust(w) + "   " + " ".join(cell(s, m).rjust(11) for m in measures))


def _fmt_pct(x):
    return "n/a" if x != x else f"{x * 100:.0f}%"


def _fmt_num(x):
    return "n/a" if x != x else f"{x:.1f}"


if __name__ == "__main__":
    main()
