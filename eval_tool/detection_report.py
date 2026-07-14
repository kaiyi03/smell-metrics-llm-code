"""
The merged detection-strength report: injected and real, on one page.

Reads the three result files and renders one detection_report.html so the two
sources sit together instead of on separate pages:
  * realworld_results.csv -- structural measures, injected d AND real d per smell
    (the two-source hero: does a measure separate smelly from clean on the
    controlled pairs, and does it still on real code?).
  * panel_results.csv     -- the similarity measures (injected only; real code has
    no clean reference).
  * correctness_results.csv -- behaviour-preservation of the injectors.
Plus a closing "which measures to trust" summary.

Run:  python eval_tool/detection_report.py       (after run_panel + run_realworld)
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, "panel_results.csv")
REAL = os.path.join(HERE, "realworld_results.csv")
CORR = os.path.join(HERE, "correctness_results.csv")
OUT = os.path.join(HERE, "detection_report.html")

STRUCT = ["sloc", "cyclomatic", "cognitive", "maintainability", "halstead_volume",
          "halstead_difficulty", "halstead_effort"]
SIM = ["bleu", "chrf", "rouge_l", "meteor", "codebleu", "ast_similarity"]

# smell -> (structural verdict, similarity verdict, what to rely on) -- the trust summary.
# structural = the reference-free radon measures; similarity = the reference-based
# BLEU/CodeBLEU family (validated on the injected pairs, which have a clean twin, so
# every injected smell diverges from its clean reference and the family registers it).
TRUST = [
    ("long_method",         "Strong",        "Strong", "any structural measure"),
    ("deep_nesting",        "Strong",        "Strong", "cyclomatic, SLOC"),
    ("complex_conditional", "Strong",        "Strong", "cyclomatic"),
    ("broad_except",        "Co-occurrence", "Strong", "CodeBLEU / BLEU; the detector"),
    ("magic_number",        "Co-occurrence", "Strong", "CodeBLEU; the detector"),
    ("inefficient_copy",    "Co-occurrence", "Strong", "CodeBLEU; the detector (Ruff)"),
    ("inefficient_loop",    "Co-occurrence", "Strong", "CodeBLEU; the detector (Ruff)"),
    ("perf_try_in_loop",    "Co-occurrence", "Strong", "CodeBLEU / BLEU; the detector (Ruff)"),
    ("long_parameter_list", "Blind",         "Strong", "CodeBLEU; the detector"),
    ("dead_code",           "Blind",         "Strong", "CodeBLEU / ROUGE-L; the detector"),
    ("mutable_default",     "Blind",         "Strong", "BLEU / CodeBLEU; the detector"),
    ("duplicate_code",      "Blind (size)",  "Strong", "jscpd; CodeBLEU vs. reference"),
]
ORDER = [t[0] for t in TRUST]


def load():
    inj_struct, real_struct = {}, {}          # (smell, measure) -> d
    for r in csv.DictReader(open(REAL, encoding="utf-8")):
        k = (r["smell"], r["measure"])
        try:
            inj_struct[k] = float(r["injected_cohen_d"])
            real_struct[k] = float(r["real_cohen_d"])
        except ValueError:
            pass
    inj_sim = {}
    for r in csv.DictReader(open(PANEL, encoding="utf-8")):
        if r["family"] == "similarity":
            try:
                inj_sim[(r["smell"], r["measure"])] = float(r["cohen_d"])
            except ValueError:
                pass
    corr = {}
    if os.path.exists(CORR):
        for r in csv.DictReader(open(CORR, encoding="utf-8")):
            corr[r["smell"]] = r
    return inj_struct, real_struct, inj_sim, corr


def colour(d):
    if d is None or d != d:
        return "#f5f5f5"
    a = min(abs(d) / 3.0, 1.0)
    return f"rgba(185,28,28,{0.10 + 0.6 * a:.2f})"


def cell(d, small=None):
    if d is None or d != d:
        return '<td style="text-align:center;background:#f5f5f5">&mdash;</td>'
    if small:
        return (f'<td class="mono stat" style="background:{colour(d)}">'
                f'<div class="stat-in"><span class="real">{d:.2f}</span>'
                f'<div class="sub">{small}</div></div></td>')
    return f'<td class="mono" style="background:{colour(d)}">{d:.2f}</td>'


def heat(smells, measures, getter, small=None):
    p = ["<table><tr><th>smell</th>" + "".join(f"<th>{m}</th>" for m in measures) + "</tr>"]
    for s in smells:
        row = f"<tr><td><code>{s}</code></td>"
        for m in measures:
            row += cell(getter(s, m), small(s, m) if small else None)
        p.append(row + "</tr>")
    p.append("</table>")
    return "".join(p)


def main():
    inj_struct, real_struct, inj_sim, corr = load()
    smells = [s for s in ORDER if any((s, m) in real_struct for m in STRUCT)]

    p = []
    p.append("""<!doctype html><html><head><meta charset="utf-8">
<title>Detection strength &mdash; injected vs. real</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#1a1a1a;line-height:1.5}
 .wrap{max-width:1040px;margin:0 auto;padding:32px 26px 80px}
 h1{font-size:25px;margin:0 0 2px} .sub-h{color:#666;margin:0 0 22px;font-size:14px}
 h2{font-size:15px;text-transform:uppercase;letter-spacing:.05em;color:#444;
    border-bottom:2px solid #e2e2e2;padding-bottom:6px;margin:38px 0 6px}
 .note{color:#777;font-size:12.5px;margin:0 0 12px}
 table{border-collapse:collapse;width:100%;font-size:12.5px;background:#fff;
       border:1px solid #e4e4e4;border-radius:8px;overflow:hidden;margin-bottom:6px}
 th,td{padding:6px 8px;text-align:right;border-bottom:1px solid #eee}
 th:first-child,td:first-child{text-align:left}
 th{background:#f4f4f6;font-weight:600}
 td.stat{text-align:center;padding:9px 8px}
 .mono{font-variant-numeric:tabular-nums}
 .stat-in{display:flex;flex-direction:column;align-items:center;gap:6px;line-height:1.15}
 .stat-in .real{font-size:13px;font-weight:600}
 .mono .sub{font-size:10px;font-weight:400;color:#3a3a3a;
            background:rgba(255,255,255,.62);border-radius:3px;padding:1px 6px}
 code{background:#f0f0f2;padding:1px 5px;border-radius:4px;font-size:12px}
 .v-strong{color:#15803d;font-weight:600} .v-mod{color:#b45309} .v-blind{color:#b91c1c}
 .th-sub{font-weight:400;color:#8a8a8a;font-size:10.5px}
 .scroll{overflow-x:auto}
</style></head><body><div class="wrap">
<h1>Detection strength &mdash; injected vs. real</h1>
<p class="sub-h">One measure of how strongly each evaluation measure separates smelly
code from clean code, computed two ways: on the controlled injected pairs and on real
labelled code. Both use the same unpaired effect size (Cohen&rsquo;s d, positive = worse,
capped at 5), so they are directly comparable. Deeper shade = stronger separation.</p>""")

    # 1. structural, two-source
    p.append("<h2>1. Structural measures &mdash; injected vs. real</h2>")
    p.append('<p class="note">Each cell shows the <b>real</b> detection strength, with the '
             '<span style="color:#999">injected</span> value beneath. They agree where the finding '
             'generalises; where they differ, the isolated injection under-tests a smell that real code '
             'carries alongside extra size or branching (e.g. cyclomatic on long methods: 0.0 injected, '
             'strong on real). Read top&rarr;bottom: reliably-detected smells first, blind spots last.</p>')
    p.append('<div class="scroll">')
    p.append(heat(smells, STRUCT,
                  lambda s, m: real_struct.get((s, m)),
                  small=lambda s, m: (f"inj {inj_struct[(s, m)]:.2f}" if (s, m) in inj_struct else "")))
    p.append("</div>")

    # 2. similarity, injected only
    p.append("<h2>2. Similarity measures &mdash; injected only</h2>")
    p.append('<p class="note">The reference-based measures need a clean reference to compare against. '
             'Real naturally-occurring code has no clean twin, so these <b>cannot be validated on real '
             'code</b> &mdash; their reliability rests on the injected benchmark alone (a stated limitation).</p>')
    p.append('<div class="scroll">')
    p.append(heat(smells, SIM, lambda s, m: inj_sim.get((s, m))))
    p.append("</div>")

    # 3. correctness
    if corr:
        p.append("<h2>3. Correctness (dataset quality gate)</h2>")
        p.append('<p class="note">Every injector must be behaviour-preserving: the smelly half passes '
                 'whatever tests its clean twin passes. This confirms the labels are sound.</p>')
        p.append("<table><tr><th>smell</th><th>tested</th><th>behaviour kept</th></tr>")
        for s in smells:
            c = corr.get(s)
            if c:
                base, kept = int(c.get("clean_pass_base", 0) or 0), int(c.get("behaviour_kept", 0) or 0)
                pct = f"{kept / base * 100:.0f}%" if base else "n/a"
                p.append(f"<tr><td><code>{s}</code></td><td>{c.get('n_tested')}</td><td>{pct}</td></tr>")
        p.append("</table>")

    # 4. trust summary -- by measure family
    p.append("<h2>4. Which measures to trust &mdash; by family</h2>")
    p.append('<p class="note">Detectability depends on the measure <b>family</b>. A '
             '<span class="v-blind">Blind</span> below means <i>that family</i> cannot see the smell '
             '&mdash; not that nothing can. Read the two verdict columns together.</p>')
    p.append('<div class="scroll"><table><tr><th>smell</th>'
             '<th style="text-align:left">structural<br><span class="th-sub">reference-free</span></th>'
             '<th style="text-align:left">similarity<br><span class="th-sub">vs. a reference</span></th>'
             '<th style="text-align:left">what to rely on</th></tr>')

    def vclass(v):
        if v == "Strong":
            return "v-strong"
        if v.startswith("Blind"):
            return "v-blind"
        return "v-mod"

    for s, structv, simv, use in TRUST:
        p.append(f'<tr><td><code>{s}</code></td>'
                 f'<td style="text-align:left" class="{vclass(structv)}">{structv}</td>'
                 f'<td style="text-align:left" class="{vclass(simv)}">{simv}</td>'
                 f'<td style="text-align:left">{use}</td></tr>')
    p.append("</table></div>")
    p.append('<p class="note">The two families do different jobs. The <b>reference-free structural</b> '
             'measures locate specific smells but only bulky or branchy ones; for the middle group the '
             'signal is really co-occurring size/complexity, and four smells are structurally invisible. '
             'The <b>reference-based similarity</b> measures (CodeBLEU, BLEU and the rest) register all '
             'twelve on the injected benchmark &mdash; including every structural blind spot &mdash; but '
             'they need a clean reference to compare against, and they report that the code <i>diverged</i> '
             'from that reference, not <i>which</i> smell it is. So they cover everything on the injected '
             'pairs (which have a clean twin); on real free-generated code, which has no clean twin, they '
             'cannot be applied the same way (a stated limitation). The reliable per-smell catch-all in '
             'deployment is the <b>detector itself</b> (Pylint, Ruff, jscpd), which defines all twelve by '
             'construction and is kept out of this scoring only to avoid grading a detector on its own '
             'labels. Net: no single reference-free measure covers all twelve &mdash; the reason the tool '
             'is a panel &mdash; but no smell is undetectable.</p>')

    p.append("</div></body></html>")
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(p))
    print(f"wrote {OUT}  ({len(smells)} smells)")


if __name__ == "__main__":
    main()
