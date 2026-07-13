"""
Real-world validation layer (the benchmark's second source).

run_panel.py evaluates the measures on the INJECTED benchmark -- synthetic
clean/smelly pairs where we control exactly which smell is present. This is the
complement the project scope calls for: it runs the same structural measures on
REAL, naturally-occurring labelled code, to check whether the findings generalise
beyond the artificial samples ("show agreement with the labels in the reused
validated dataset").

Data (both real, both from the CodeSmellData 2.0 GitHub corpus, so same
distribution):
  * smelly side -- reused.jsonl: real methods Pylint flagged with each smell
    (CodeSmellData 1.0/2.0), plus PySmell's human-labelled examples.
  * clean side  -- realworld_clean.jsonl: real methods from the SAME corpus that
    carry NONE of our tracked smells (built by build_clean_baseline.py). This is
    the baseline the pre-labelled files lack -- they are all-positive.

For every smell we compute a real detection strength: the UNPAIRED Cohen's d
between the real smelly group and the real-clean pool, per structural measure,
oriented so positive = worse and capped at +/-5. It is shown next to the injected
(paired) value, so "does the measure separate smelly from clean on synthetic
pairs, and does it still separate them on real code?" reads directly.

Only the reference-free STRUCTURAL measures apply -- real code has no clean twin,
so the similarity measures cannot run.

Interpretation: real d reflects the measure's response to the smell PLUS whatever
co-occurs with it in the wild (real long methods also tend to be complex), whereas
injected d isolates the smell. Comparing the two is the point: agreement means the
finding generalises; a big drop means injection exaggerated the measure; injected
~0 but real >0 means injection created a false blind spot the real data corrects.

For long_method and long_parameter_list, PySmell also provides human-labelled
clean negatives, used as an independent cross-check on the pooled baseline.

Run:  python eval_tool/run_realworld.py            (auto-switches to the venv)
"""

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

import csv
import json
import math
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
REUSED = os.path.join(ROOT, "smell_injection", "reused.jsonl")
CLEAN = os.path.join(ROOT, "smell_injection", "realworld_clean.jsonl")
MINED = os.path.join(ROOT, "smell_injection", "realworld_smelly.jsonl")
INJECTED_CSV = os.path.join(HERE, "panel_results.csv")
OUT_CSV = os.path.join(HERE, "realworld_results.csv")
OUT_HTML = os.path.join(HERE, "realworld_report.html")
D_CAP = 5.0
PYSMELL_SMELLS = ("long_method", "long_parameter_list")   # the two with human clean negatives

from measures import PANEL                                    # noqa: E402
STRUCT = [m for m in PANEL if not m.needs_ref]                # reference-free only


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_data():
    smelly = defaultdict(list)                    # smell -> [code]
    pysmell_clean = defaultdict(list)             # smell -> [code]  (human negatives)
    for r in load_jsonl(REUSED):
        if r.get("label") == "yes":
            smelly[r["smell"]].append(r["code"])
        elif r.get("label") == "no":
            pysmell_clean[r["smell"]].append(r["code"])
    if os.path.exists(MINED):                     # real examples of the smells the reused
        for r in load_jsonl(MINED):               # (Pylint-era) datasets don't cover
            if r.get("label") == "yes":
                smelly[r["smell"]].append(r["code"])
    clean_pool = [r["code"] for r in load_jsonl(CLEAN)] if os.path.exists(CLEAN) else []
    return smelly, clean_pool, pysmell_clean


def load_injected_d():
    out = {}
    if os.path.exists(INJECTED_CSV):
        for row in csv.DictReader(open(INJECTED_CSV, encoding="utf-8")):
            try:
                out[(row["smell"], row["measure"])] = float(row["cohen_d"])
            except (KeyError, ValueError):
                pass
    return out


def vals(codes, m):
    return [v for v in (m.fn(c) for c in codes) if v is not None]


def cohens_d(smelly, clean, worse):
    """Unpaired Cohen's d, oriented positive = worse, capped at +/-D_CAP."""
    if len(smelly) < 2 or len(clean) < 2:
        return float("nan")
    ms, mc = statistics.fmean(smelly), statistics.fmean(clean)
    n1, n2 = len(smelly), len(clean)
    s1, s2 = statistics.stdev(smelly), statistics.stdev(clean)
    pooled = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        d = 0.0 if ms == mc else math.copysign(D_CAP, ms - mc)
    else:
        d = (ms - mc) / pooled
    d = d if worse == "up" else -d
    return max(-D_CAP, min(D_CAP, d))


def main():
    smelly, clean_pool, pysmell_clean = load_data()
    injected_d = load_injected_d()
    smells = sorted(smelly)

    if len(clean_pool) < 2:
        raise SystemExit(f"clean baseline missing/too small ({len(clean_pool)}). "
                         f"Run: python smell_injection/build_clean_baseline.py")

    print(f"real smelly: {sum(len(v) for v in smelly.values())} across {len(smells)} smells")
    print(f"real-clean pool: {len(clean_pool)} methods (CodeSmellData 2.0 raw, no tracked smell)")
    print(f"structural measures only (similarity needs a clean twin)\n")

    # measure values, computed once per group
    clean_v = {m.name: vals(clean_pool, m) for m in STRUCT}
    smelly_v = {(s, m.name): vals(smelly[s], m) for s in smells for m in STRUCT}
    pys_v = {(s, m.name): vals(pysmell_clean[s], m) for s in PYSMELL_SMELLS
             for m in STRUCT if pysmell_clean.get(s)}

    rows_out = []
    print(f"Real detection strength (real smelly vs real-clean pool), injected value for comparison:")
    for s in smells:
        print(f"  {s}  ({len(smelly[s])} real smelly vs {len(clean_pool)} clean)")
        print(f"    {'measure':22}{'injected d':>12}{'real d':>10}{'clean med':>11}{'smelly med':>12}")
        for m in STRUCT:
            sv, cv = smelly_v[(s, m.name)], clean_v[m.name]
            d = cohens_d(sv, cv, m.worse)
            inj = injected_d.get((s, m.name), float("nan"))
            cm = statistics.median(cv) if cv else float("nan")
            smd = statistics.median(sv) if sv else float("nan")
            pys = cohens_d(sv, pys_v[(s, m.name)], m.worse) if (s, m.name) in pys_v else float("nan")
            print(f"    {m.name:22}{inj:>12.2f}{d:>10.2f}{cm:>11.1f}{smd:>12.1f}")
            rows_out.append([s, m.name, len(smelly[s]), len(clean_pool),
                             f"{inj:.3f}", f"{d:.3f}",
                             "" if pys != pys else f"{pys:.3f}", f"{cm:.3f}", f"{smd:.3f}"])

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smell", "measure", "n_smelly", "n_clean", "injected_cohen_d",
                    "real_cohen_d", "real_d_vs_pysmell_negatives", "clean_median", "smelly_median"])
        w.writerows(rows_out)

    write_html(smells, smelly, clean_pool, pysmell_clean, smelly_v, clean_v, pys_v, injected_d)
    print(f"\nwrote {os.path.basename(OUT_CSV)} and {os.path.basename(OUT_HTML)}")


def _colour(d):
    if d != d:
        return "#f5f5f5"
    a = min(abs(d) / 3.0, 1.0)
    return f"rgba(185,28,28,{0.10 + 0.6 * a:.2f})"


def write_html(smells, smelly, clean_pool, pysmell_clean, smelly_v, clean_v, pys_v, injected_d):
    import statistics as st
    p = []
    p.append(f"""<!doctype html><html><head><meta charset="utf-8">
<title>Real-world validation &mdash; structural measures</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#1a1a1a;line-height:1.5}}
 .wrap{{max-width:1040px;margin:0 auto;padding:32px 26px 80px}}
 h1{{font-size:24px;margin:0 0 2px}} .sub{{color:#666;margin:0 0 22px;font-size:14px}}
 h2{{font-size:14px;text-transform:uppercase;letter-spacing:.06em;color:#555;
     border-bottom:2px solid #e2e2e2;padding-bottom:6px;margin:34px 0 6px}}
 .note{{color:#777;font-size:12.5px;margin:0 0 12px}}
 table{{border-collapse:collapse;width:100%;font-size:13px;background:#fff;
        border:1px solid #e4e4e4;border-radius:8px;overflow:hidden;margin-bottom:8px}}
 th,td{{padding:6px 9px;text-align:right;border-bottom:1px solid #eee}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#f4f4f6;font-weight:600}}
 code{{background:#f0f0f2;padding:1px 5px;border-radius:4px;font-size:12px}}
 .prov{{font-size:12px;color:#888}}
 .mono{{font-variant-numeric:tabular-nums}}
</style></head><body><div class="wrap">
<h1>Real-world validation &mdash; structural measures</h1>
<p class="sub">The benchmark's second source. The structural measures run on real,
naturally-occurring code: methods Pylint flagged with each smell (CodeSmellData
1.0/2.0, PySmell) versus a pool of {len(clean_pool)} real <b>clean</b> methods from
the same GitHub corpus. Detection strength = unpaired Cohen&rsquo;s d (smelly vs
clean), shown against the injected value (now the same statistic). The similarity
measures need a clean reference, so they do not apply to real code. duplicate_code
appears as real clone-pairs concatenated into one snippet -- the same doubled shape
the injector builds -- so its structural "signal" on both sources is only the size
of the second copy, not the duplication itself; a clone detector (jscpd), not the
structural measures, is what actually finds it.</p>""")

    # headline matrix: real d per smell x measure
    p.append("<h2>Real detection strength (smell &times; measure)</h2>")
    p.append('<p class="note">Real smelly code vs the real-clean pool. Positive = worse; capped &plusmn;5. '
             'Deeper shade = stronger separation.</p>')
    p.append("<table><tr><th>smell</th>" + "".join(f"<th>{m.name}</th>" for m in STRUCT) + "</tr>")
    for s in smells:
        cells = ""
        for m in STRUCT:
            d = cohens_d(smelly_v[(s, m.name)], clean_v[m.name], m.worse)
            cells += f'<td class="mono" style="background:{_colour(d)}">{"" if d != d else f"{d:.2f}"}</td>'
        p.append(f"<tr><td><code>{s}</code></td>{cells}</tr>")
    p.append("</table>")

    # generalisation: injected vs real, per smell
    p.append("<h2>Injected vs real, by smell</h2>")
    p.append('<p class="note">Where they agree, the finding generalises. A big drop means the injection '
             'exaggerated the measure; injected &asymp;0 but real &gt;0 means the injection created a false '
             'blind spot the real data corrects.</p>')
    for s in smells:
        p.append(f'<p style="margin:14px 0 4px"><b>{s}</b> '
                 f'<span class="prov">({len(smelly[s])} real smelly vs {len(clean_pool)} clean'
                 + (f'; PySmell cross-check on {len(pysmell_clean[s])} human negatives' if pysmell_clean.get(s) else "")
                 + ')</span></p>')
        has_pys = bool(pysmell_clean.get(s))
        p.append("<table><tr><th>measure</th><th>injected d</th><th>real d</th>"
                 + ("<th>real d (PySmell neg.)</th>" if has_pys else "")
                 + "<th>clean median</th><th>smelly median</th></tr>")
        for m in STRUCT:
            sv, cv = smelly_v[(s, m.name)], clean_v[m.name]
            d = cohens_d(sv, cv, m.worse)
            inj = injected_d.get((s, m.name), float("nan"))
            cm, smd = (st.median(cv) if cv else float("nan")), (st.median(sv) if sv else float("nan"))
            pys_cell = ""
            if has_pys:
                pd = cohens_d(sv, pys_v.get((s, m.name), []), m.worse)
                pys_cell = f'<td class="mono">{"" if pd != pd else f"{pd:.2f}"}</td>'
            p.append(f'<tr><td>{m.name}</td>'
                     f'<td class="mono">{"" if inj != inj else f"{inj:.2f}"}</td>'
                     f'<td class="mono" style="background:{_colour(d)}">{"" if d != d else f"{d:.2f}"}</td>'
                     f'{pys_cell}'
                     f'<td class="mono">{"" if cm != cm else f"{cm:.1f}"}</td>'
                     f'<td class="mono">{"" if smd != smd else f"{smd:.1f}"}</td></tr>')
        p.append("</table>")

    # provenance
    p.append("<h2>Data provenance</h2><table><tr><th>smell</th><th>real smelly</th>"
             "<th>PySmell human negatives</th></tr>")
    for s in smells:
        p.append(f"<tr><td><code>{s}</code></td><td>{len(smelly[s])}</td>"
                 f"<td>{len(pysmell_clean.get(s, []))}</td></tr>")
    p.append(f"<tr><td><b>clean pool (shared)</b></td><td>&mdash;</td><td>{len(clean_pool)}</td></tr>")
    p.append("</table>")
    p.append('<p class="prov">Smelly: CodeSmellData 1.0/2.0 (Pylint-labelled) + PySmell (human). '
             'Clean pool: real methods from the CodeSmellData 2.0 raw corpus that carry no tracked smell '
             '(build_clean_baseline.py). Not circular: detectors define the clean/smelly split; the '
             'independent structural measures are what is tested against it.</p>')

    p.append("</div></body></html>")
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write("\n".join(p))


if __name__ == "__main__":
    main()
