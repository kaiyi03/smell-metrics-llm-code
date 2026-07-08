"""
Render the panel results as a readable, colour-coded HTML report.

The CSV is the full record; this is the at-a-glance view. Two heatmaps:

  structural  cell = hit-rate: how often the smell moved the measure the worse
              way. Greener = the measure detects this smell more reliably.
  similarity  cell = how similar the smelly code still is to the clean reference
              (100 = identical). Greener = the smell perturbed the code more, i.e.
              the measure noticed it more.

Hover any cell for the underlying numbers (n, deltas, medians). The file is
self-contained (inline CSS) -- just open it in a browser.
"""

import html


def _bg(strength):
    """strength in [0,1] -> pale grey (weak/blind) to green (strong)."""
    strength = max(0.0, min(1.0, strength))
    r = int(245 + (46 - 245) * strength)
    g = int(245 + (160 - 245) * strength)
    b = int(245 + (67 - 245) * strength)
    return f"rgb({r},{g},{b})"


def _fg(strength):
    return "#fff" if strength > 0.55 else "#111"


def _cell(text, strength, tip):
    return (f'<td style="background:{_bg(strength)};color:{_fg(strength)}" '
            f'title="{html.escape(tip)}">{html.escape(text)}</td>')


def _struct_strength(c):
    hr = c["hit_rate"]
    return hr if hr == hr else 0.0                       # hr==hr rejects NaN


def _sim_strength(c):
    sm = c["smelly_median"]
    return max(0.0, (100 - sm) / 100) if sm == sm else 0.0


def _heatmap(smells, measures, cells, kind):
    head = "".join(f"<th>{html.escape(m)}</th>" for m in measures)
    rows = []
    for s in smells:
        tds = [f'<th class="rowh">{html.escape(s)}</th>']
        for m in measures:
            c = cells[(s, m)]
            if kind == "structural":
                strength = _struct_strength(c)
                text = f"{strength * 100:.0f}%"
                tip = (f"{s} / {m}\n"
                       f"moved worse in {strength * 100:.0f}% of {c['n']} pairs\n"
                       f"median change per pair {c['delta_median']:+.2f} "
                       f"(mean {c['delta_mean']:+.2f})\n"
                       f"median clean {c['clean_median']:.2f} → "
                       f"median smelly {c['smelly_median']:.2f}")
            else:
                sm = c["smelly_median"]
                strength = _sim_strength(c)
                text = f"{sm:.0f}" if sm == sm else "n/a"
                tip = (f"{s} / {m}\n"
                       f"smelly code scores {sm:.1f}/100 for similarity to its clean twin\n"
                       f"(identical code scores 100)\n"
                       f"the smell cost {100 - sm:.1f} points across {c['n']} pairs")
            tds.append(_cell(text, strength, tip))
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="scroll"><table>'
            f'<tr><th class="corner">smell \\ measure</th>{head}</tr>'
            f'{"".join(rows)}</table></div>')


def _d_strength(c):
    d = c.get("cohen_d", float("nan"))
    if d != d:                                          # NaN
        return 0.0
    return max(0.0, min(1.0, d / 3.0))                  # only "worse" (positive) greens up


def _master(smells, measures, cells):
    head = "".join(f"<th>{html.escape(m)}</th>" for m in measures)
    rows = []
    for s in smells:
        tds = [f'<th class="rowh">{html.escape(s)}</th>']
        for m in measures:
            c = cells[(s, m)]
            d = c.get("cohen_d", float("nan"))
            rd = 0.0 if (d == d and abs(d) < 0.05) else d       # avoid a "-0.0" display
            text = "n/a" if d != d else f"{rd:.1f}"
            tip = (f"{s} / {m}\n"
                   f"detection strength = {rd:.2f}  (positive = moved the worse way)\n"
                   f"hit-rate {c['hit_rate'] * 100:.0f}%  |  "
                   f"median change per pair {c['delta_median']:+.2f}")
            tds.append(_cell(text, _d_strength(c), tip))
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="scroll"><table>'
            f'<tr><th class="corner">smell \\ measure</th>{head}</tr>'
            f'{"".join(rows)}</table></div>')


def _correctness_section(correctness):
    if not correctness:
        return ""
    rows = []
    for smell in sorted(correctness):
        c = correctness[smell]
        n, base, kept = c["n_tested"], c["clean_pass_base"], c["behaviour_kept"]
        cp = c["clean_pass"] / n * 100 if n else float("nan")
        sp = c["smelly_pass"] / n * 100 if n else float("nan")
        kp = kept / base * 100 if base else float("nan")
        strength = 1.0 if (kp == kp and kp >= 99.5) else 0.0
        kept_cell = _cell(f"{kp:.0f}%" if kp == kp else "n/a", strength,
                          f"{smell}: {kept} of {base} clean-passing pairs kept behaviour")
        rows.append(f'<tr><th class="rowh">{html.escape(smell)}</th>'
                    f'<td>{n}</td><td>{cp:.0f}%</td><td>{sp:.0f}%</td>{kept_cell}</tr>')
    return ('<h2>Correctness &mdash; does the code still run? (execution)</h2>'
            '<p class="muted">Each snippet is executed against its task&#39;s real tests. Here it '
            'checks the injectors preserved behaviour; the same check scores the model&#39;s output '
            'later. CodeSearchNet-sourced samples have no tests and are omitted. &ldquo;Behaviour '
            'kept&rdquo; = of the pairs whose clean twin passes, how many the smelly twin also '
            'passes.</p>'
            '<div class="scroll"><table>'
            '<tr><th class="corner">smell</th><th>tested</th><th>clean pass</th>'
            '<th>smelly pass</th><th>behaviour kept</th></tr>'
            f'{"".join(rows)}</table></div>')


def write_html(cells, smells, structural, similarity, out_path, n_pairs, correctness=None):
    # auto blind-spot notes (kept within a family so the comparison is fair)
    struct_blind = [s for s in smells
                    if max(_struct_strength(cells[(s, m)]) for m in structural) < 0.5]
    sim_blind = [s for s in smells
                 if min(cells[(s, m)]["smelly_median"] for m in similarity) > 90]

    def note(items, msg_none, msg_some):
        if not items:
            return f'<p class="note ok">{msg_none}</p>'
        return f'<p class="note warn">{msg_some}: <b>{", ".join(items)}</b></p>'

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Measure panel report</title>
<style>
  body {{ font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif;
         color: #111; max-width: 1000px; margin: 30px auto; padding: 0 16px; }}
  h1 {{ font-size: 22px; margin-bottom: 2px; }}
  h2 {{ font-size: 17px; margin: 28px 0 4px; }}
  .muted {{ color: #666; }}
  .guide {{ background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 8px;
            padding: 12px 16px; margin: 16px 0; }}
  .guide p {{ margin: 6px 0; }}
  table {{ border-collapse: collapse; font-size: 13px; }}
  .scroll {{ overflow-x: auto; }}
  th, td {{ border: 1px solid #d0d7de; padding: 6px 9px; text-align: center;
            font-variant-numeric: tabular-nums; }}
  th {{ background: #f0f2f4; }}
  .corner {{ text-align: left; }}
  .rowh {{ text-align: left; background: #f0f2f4; white-space: nowrap; }}
  td {{ cursor: help; }}
  .legend {{ display: flex; align-items: center; gap: 8px; margin: 8px 0; font-size: 12px; }}
  .bar {{ height: 14px; width: 220px; border: 1px solid #d0d7de;
          background: linear-gradient(90deg, rgb(245,245,245), rgb(46,160,67)); }}
  .note {{ margin: 8px 0; padding: 8px 12px; border-radius: 6px; font-size: 13px; }}
  .warn {{ background: #fff8e6; border: 1px solid #f0d48a; }}
  .ok {{ background: #eef8f0; border: 1px solid #bfe3c6; }}
</style></head><body>

<h1>Measure panel &mdash; validation on the injected dataset</h1>
<p class="muted">{n_pairs} clean/smelly pairs &middot; each measure computed on both
sides &middot; the tools that made the labels (Pylint / Ruff / jscpd) are deliberately
excluded &mdash; this is the independent panel.</p>

<div class="guide">
  <p><b>Green</b> = the measure notices this smell. <b>Pale</b> = it's blind to it.</p>
  <p><b>Structural</b> cells show the <b>hit-rate</b>: % of pairs where injecting the
     smell moved the measure the worse way.</p>
  <p><b>Similarity</b> cells show <b>how similar the smelly code still is to its clean
     twin</b> (100 = identical; lower = the smell changed the code more, so the measure noticed).</p>
  <p class="muted">Hover any cell for the exact numbers. In the two family tables,
     read a measure <i>down its column</i> (across smells); don't compare raw numbers
     <i>across</i> different measures &mdash; they're different formulas. The master
     table below fixes that by putting every measure on one scale.</p>
  <div class="legend"><span>blind</span><div class="bar"></div><span>strong</span></div>
</div>

<h2>Overall &mdash; detection strength (one scale for every measure)</h2>
<p class="muted">The one place you <b>can</b> compare measures head-to-head. Each cell is a
   <b>detection-strength score</b>: how cleanly the measure tells a smelly function from its
   clean twin, on a single scale. <b>0</b> = can't tell them apart; <b>~1</b> = a clear gap;
   <b>2+</b> = very strong; <b>5</b> = capped (the change was so consistent the two barely
   overlap). Positive means the smell made the measure worse. Read <i>down a column</i> to
   profile a measure, <i>across a row</i> to see which measure best catches a smell.</p>
{_master(smells, structural + similarity, cells)}

<h2>Structural quality &mdash; reference-free measures</h2>
{_heatmap(smells, structural, cells, "structural")}
{note(struct_blind,
      "Every smell is caught by at least one structural measure.",
      "No structural measure reliably catches (all under 50%)")}
<p class="muted"><b>Note on the maintainability index:</b> it is the odd one out &mdash;
   higher is better, so a smell pushes it <i>down</i>. It is the only structural measure
   where "worse" means the number falls, which is why its change-per-pair is negative in
   the hover tooltips.</p>

<h2>Similarity to the clean reference &mdash; reference-based measures</h2>
{_heatmap(smells, similarity, cells, "similarity")}
{note(sim_blind,
      "Every smell dents at least one similarity measure by 10+ points.",
      "The similarity measures barely notice (best stays above 90/100)")}

{_correctness_section(correctness)}

<p class="muted" style="margin-top:24px">Reproducible: fixed dataset, deterministic
tools. Re-run <code>python eval_tool/run_panel.py</code> to regenerate. Full numbers in
<code>panel_results.csv</code>.</p>
</body></html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
