"""
Interactive evaluation dashboard for the code-smell project.

Paste a snippet -- optionally a reference solution and a test block -- and the
tool runs the whole evaluation in one place:

  * the smell detectors (Pylint / Ruff / jscpd)  -- the operational labels
  * the structural measures (radon + cognitive complexity + profiles)
  * the similarity measures (BLEU / CodeBLEU / AST edit distance) -- needs a reference
  * correctness -- runs the code against the test block in a sandboxed subprocess

It imports the SAME measures.PANEL, build_injected.all_smells and
correctness.run_program that the offline harness uses, so the numbers match the
benchmark exactly -- the dashboard is just a front-end onto the existing tool.

Run:  .venv/Scripts/python dashboard/app.py     then open http://127.0.0.1:5000
"""

import ast
import bisect
import json
import os
import subprocess
import sys
import tempfile
import threading
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))


def _bootstrap():
    """Re-launch under the project .venv so the measure libraries import, no matter
    how the app was started. Reloader is off (below) so this only happens once."""
    venv = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    if not os.path.exists(venv):                       # linux / ARC layout
        venv = os.path.join(ROOT, ".venv", "bin", "python")
    if os.path.exists(venv) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv).lower():
        print(f"[setup] switching to project venv:\n        {venv}\n")
        raise SystemExit(subprocess.run([venv, os.path.abspath(__file__), *sys.argv[1:]]).returncode)


_bootstrap()

sys.path.insert(0, os.path.join(ROOT, "eval_tool"))
sys.path.insert(0, os.path.join(ROOT, "smell_injection"))

from flask import Flask, request, render_template_string   # noqa: E402
from measures import PANEL                                  # noqa: E402
import build_injected as BI                                 # noqa: E402
from correctness import run_program                         # noqa: E402

try:
    from detection_report import TRUST                      # noqa: E402
except Exception:
    TRUST = []
TRUST_MAP = {row[0]: row for row in TRUST}   # smell -> (smell, structural, similarity, use)

STRUCT = [m for m in PANEL if not m.needs_ref]
SIM = [m for m in PANEL if m.needs_ref]

# Clean-code baseline (percentile curves per structural measure), precomputed by
# build_baselines.py from the real-clean pool. Lets us say where a reading sits.
try:
    with open(os.path.join(HERE, "baselines.json"), encoding="utf-8") as f:
        BASELINES = json.load(f).get("clean", {})
except Exception:
    BASELINES = {}
PROFILE = {"comment_density", "api_calls"}   # profiling measures: show percentile, no good/bad tier

# Smelly-side reference: real clean vs smelly medians per (smell, measure), from the
# real-world results table. Lets a detected smell show "clean -> you -> smelly".
SMELLY = {}
try:
    import csv as _csv
    _rw = os.path.join(ROOT, "eval_tool", "realworld_results.csv")
    for _r in _csv.DictReader(open(_rw, encoding="utf-8")):
        try:
            SMELLY[(_r["smell"], _r["measure"])] = {
                "clean": float(_r["clean_median"]), "smelly": float(_r["smelly_median"]),
                "d": float(_r["real_cohen_d"])}
        except (ValueError, KeyError):
            pass
except Exception:
    pass

# Pre-filled on first load so the first "Evaluate" visibly detects smells -- this
# example trips three low-threshold detectors (mutable default, unused variable,
# broad except); the reference and tests light up the similarity and correctness panels.
EXAMPLE_CODE = """def scale(values, factor, seen=[]):
    result = []
    unused = 0
    for v in values:
        try:
            result.append(v * factor)
        except Exception:
            pass
    seen.append(len(result))
    return result
"""
EXAMPLE_REF = """def scale(values, factor):
    return [v * factor for v in values]
"""
EXAMPLE_TESTS = """assert scale([1, 2, 3], 2) == [2, 4, 6]
assert scale([], 5) == []
"""

app = Flask(__name__)


def _fmt(v, dp):
    return "&mdash;" if v is None else f"{v:.{dp}f}"


def _run_detector(cmd):
    """Run a detector subprocess and parse its JSON. Returns (messages, error).
    A non-zero exit is NOT an error for these tools -- pylint uses its exit code as
    a bitmask and ruff returns 1 when it finds issues; the real failure is producing
    no parseable JSON or failing to launch at all."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as e:
        return None, f"could not launch ({type(e).__name__}: {e})"
    try:
        return json.loads(proc.stdout or "[]"), None
    except json.JSONDecodeError:
        detail = (proc.stderr or proc.stdout or "no output").strip().replace("\n", " ")
        return None, f"exit {proc.returncode}: {detail[:200]}"


def detect_labeled(code):
    """Detection as in build_injected.all_smells, but it also records WHERE each smell
    is (line + rule) and surfaces detector failures instead of returning a silent empty
    set. Returns (smells, locations, problems); locations maps a smell to a list of
    (line, rule, message) tuples."""
    locations, problems = {}, []
    try:                                       # unparseable code -> say so, don't look "clean"
        ast.parse(code)
    except SyntaxError as e:
        return [], {}, [f"the code has a syntax error and cannot be analysed: {e.msg} (line {e.lineno})"]

    def add(smell, line, rule, message):
        locations.setdefault(smell, []).append((line, rule, message))

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        msgs, err = _run_detector(
            [sys.executable, "-m", "pylint", path,
             "--load-plugins=pylint.extensions.magic_value",
             "--output-format=json", "--disable=all", f"--enable={BI.PYLINT_ENABLE}"])
        if err:
            problems.append(f"pylint {err}")
        else:
            for m in msgs:
                mid = m.get("message-id")
                if mid in BI.MSGID_TO_SMELL:
                    add(BI.MSGID_TO_SMELL[mid], m.get("line"), mid, (m.get("message") or "").strip())
        msgs, err = _run_detector(
            [sys.executable, "-m", "ruff", "check", path, "--isolated",
             "--select", BI.RUFF_SELECT, "--output-format=json"])
        if err:
            problems.append(f"ruff {err}")
        else:
            for m in msgs:
                rule = m.get("code")
                if rule in BI.RULE_TO_SMELL:
                    row = (m.get("location") or {}).get("row")
                    add(BI.RULE_TO_SMELL[rule], row, rule, (m.get("message") or "").strip())
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    try:
        if BI.has_duplicate(code):
            add("duplicate_code", None, "jscpd", "a duplicated block was found in the snippet")
    except Exception as e:
        problems.append(f"jscpd: {e}")

    for s in locations:                        # earliest line first
        locations[s].sort(key=lambda t: (t[0] is None, t[0]))
    return sorted(locations), locations, problems


def _baseline(name, value):
    """Where a reading sits versus clean code: (label, css) or None. label is like
    'p94 - elevated'; the percentile is against the 1000 real clean functions."""
    b = BASELINES.get(name)
    if b is None or value is None:
        return None
    p = bisect.bisect_right(b["dist"], value)            # 0..100 percentile in clean
    if name in PROFILE:                                  # profile: percentile only, no verdict
        return (f"p{p}", "b-neutral")
    if b["worse"] == "up":                               # higher = worse
        tier = ("high", "b-high") if p >= 95 else ("elevated", "b-mid") if p >= 75 else ("typical", "b-ok")
    else:                                                # lower = worse (maintainability)
        tier = ("very low", "b-high") if p <= 5 else ("low", "b-mid") if p <= 25 else ("typical", "b-ok")
    return (f"p{p} &middot; {tier[0]}", tier[1])


def evaluate(code, ref, tests, run_tests):
    """Run the panel on one snippet. Returns a dict of display-ready pieces."""
    smells, locations, problems = detect_labeled(code)
    vals = {m.name: m.fn(code) for m in STRUCT}
    structural = [(m.name, _fmt(vals[m.name], 2), m.blurb, _baseline(m.name, vals[m.name]))
                  for m in STRUCT]

    # per detected smell: clean -> you -> smelly, for the measures that actually
    # separate that smell on real code (real Cohen's d >= 1); empty = structure is
    # blind to it (rely on similarity / the detector instead).
    compare = []
    for s in smells:
        rows = []
        for m in STRUCT:
            info = SMELLY.get((s, m.name))
            if info and info["d"] is not None and info["d"] >= 1.0:
                rows.append((info["d"], m.name, _fmt(info["clean"], 2),
                             _fmt(vals[m.name], 2), _fmt(info["smelly"], 2)))
        rows.sort(reverse=True)
        compare.append((s, [(n, c, y, sm) for _, n, c, y, sm in rows[:4]]))

    similarity = None
    if ref.strip():
        similarity = [(m.name, _fmt(m.fn(code, ref), 1), m.blurb) for m in SIM]

    correctness = None
    if tests.strip() and run_tests:
        correctness = run_program(code + "\n\n" + tests)

    detections = [{"smell": s, "locs": locations.get(s, []), "trust": TRUST_MAP.get(s)}
                  for s in smells]
    return {"smells": smells, "problems": problems, "detections": detections, "compare": compare,
            "structural": structural, "similarity": similarity, "correctness": correctness,
            "has_ref": bool(ref.strip()), "has_tests": bool(tests.strip())}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation dashboard &mdash; code smells</title>
<style>
 :root{color-scheme:light dark}
 *{box-sizing:border-box}
 body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
      line-height:1.5;color:#1a1a1a;background:#fafafa}
 .wrap{max-width:1000px;margin:0 auto;padding:34px 24px 90px}
 h1{font-size:25px;margin:0 0 4px} .sub{color:#666;margin:0 0 24px;font-size:14px}
 h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#555;
    border-bottom:2px solid #e4e4e4;padding-bottom:6px;margin:30px 0 12px}
 form{background:#fff;border:1px solid #e4e4e4;border-radius:12px;padding:18px 20px;
      box-shadow:0 1px 2px rgba(0,0,0,.04)}
 label{display:block;font-size:12.5px;font-weight:600;color:#444;margin:0 0 5px}
 label .opt{font-weight:400;color:#999}
 textarea{width:100%;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
          font-size:12.5px;border:1px solid #d8d8d8;border-radius:8px;padding:10px;
          background:#fcfcfc;resize:vertical}
 .two{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
 .row{display:flex;align-items:center;justify-content:space-between;margin-top:14px;gap:14px;flex-wrap:wrap}
 .chk{font-size:12.5px;color:#555;font-weight:400;display:flex;align-items:center;gap:7px}
 button{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:10px 22px;
        font-size:14px;font-weight:600;cursor:pointer}
 button:hover{background:#1d4ed8}
 .cards{display:grid;grid-template-columns:1fr 1fr;gap:16px}
 .card{background:#fff;border:1px solid #e4e4e4;border-radius:12px;padding:16px 18px}
 .card.full{grid-column:1 / -1}
 table{border-collapse:collapse;width:100%;font-size:12.5px}
 th,td{padding:5px 8px;text-align:right;border-bottom:1px solid #efefef}
 th:first-child,td:first-child{text-align:left}
 th{color:#888;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.03em}
 td.num{font-variant-numeric:tabular-nums;font-weight:600}
 td.blurb{color:#888;font-size:11.5px;font-weight:400}
 code{background:#f0f0f2;padding:1px 5px;border-radius:4px;font-size:12px}
 .chip{display:inline-block;background:#fee2e2;color:#b91c1c;border-radius:999px;
       padding:3px 11px;font-size:12.5px;font-weight:600;margin:0 6px 6px 0}
 .clean{display:inline-block;background:#dcfce7;color:#15803d;border-radius:999px;
        padding:3px 12px;font-size:13px;font-weight:600}
 .hint{font-size:12px;color:#555;margin:2px 0 0}
 .hint b{color:#1a1a1a} .hint a{color:#2563eb;text-decoration:none} .hint a:hover{text-decoration:underline}
 .pass{background:#dcfce7;color:#15803d} .fail{background:#fee2e2;color:#b91c1c}
 .timeout{background:#fef9c3;color:#a16207}
 .badge{display:inline-block;border-radius:999px;padding:4px 14px;font-size:13px;font-weight:600}
 .muted{color:#999;font-size:12.5px}
 .warn{color:#b45309;font-size:11.5px;margin:4px 0 0}
 td.base{font-variant-numeric:tabular-nums;font-weight:600;font-size:11.5px;white-space:nowrap}
 .b-ok{color:#15803d} .b-mid{color:#b45309} .b-high{color:#b91c1c} .b-neutral{color:#6b7280}
 .detfail{color:#b91c1c;font-weight:600;font-size:13px;margin:0 0 6px}
 .detfail-list{margin:0 0 8px;padding-left:18px} .detfail-list code{font-size:11px;word-break:break-word}
 table.cmp{width:auto;margin:12px 0 0;font-size:12px}
 table.cmp caption{text-align:left;font-size:11.5px;color:#888;padding:0 0 4px;caption-side:top}
 table.cmp th,table.cmp td{padding:3px 16px 3px 0;text-align:right;border-bottom:1px solid #efefef}
 table.cmp th:first-child,table.cmp td:first-child{text-align:left}
 td.you{font-weight:700;color:#2563eb}
 .det{margin:10px 0 0}
 .det-h{margin:0;font-size:13px} .det-h .rely{color:#888;font-weight:400;font-size:12px}
 .loc{margin:1px 0 0 2px;font-size:12px;color:#555}
 .ln{display:inline-block;min-width:52px;color:#2563eb;font-weight:600;font-variant-numeric:tabular-nums}
 footer{margin-top:36px;font-size:12.5px;color:#999}
 @media (max-width:720px){.two,.cards{grid-template-columns:1fr}}
 @media (prefers-color-scheme:dark){
   body{background:#0f1115;color:#e6e6e6}.sub,.muted{color:#8a8f98}
   form,.card{background:#171a21;border-color:#2a2f3a;box-shadow:none}
   h2{color:#a8adb7;border-color:#2a2f3a}label{color:#c3c8d2}
   textarea{background:#0f1115;border-color:#2a2f3a;color:#e6e6e6}
   th,td{border-color:#242a33}td.blurb,th{color:#7a808a}
   code{background:#222732}.hint{color:#a8adb7}.hint b{color:#e6e6e6}
   td.you{color:#6ea8fe} .loc{color:#a8adb7} .ln{color:#6ea8fe} .det-h .rely{color:#7a808a}
 }
</style></head><body><div class="wrap">
 <h1>Evaluation dashboard</h1>
 <p class="sub">Paste code and run the project's evaluation panel on it: the smell
 detectors, the structural measures, similarity against a reference, and correctness
 against a test block &mdash; the same measures used on the benchmark.</p>

 <form method="post">
   <label>Code <span class="opt">&mdash; the snippet to evaluate</span></label>
   <textarea name="code" rows="12" placeholder="def solution(...):\n    ...">{{ code }}</textarea>
   <div class="two">
     <div>
       <label>Reference solution <span class="opt">&mdash; optional, enables similarity</span></label>
       <textarea name="ref" rows="7" placeholder="a known-good version to compare against">{{ ref }}</textarea>
     </div>
     <div>
       <label>Tests <span class="opt">&mdash; optional, enables correctness</span></label>
       <textarea name="tests" rows="7" placeholder="assert solution(2) == 4">{{ tests }}</textarea>
       <p class="warn">Runs your code in a subprocess. Use locally, with code you trust.</p>
     </div>
   </div>
   <div class="row">
     <label class="chk"><input type="checkbox" name="run_tests" value="1" {{ 'checked' if run_tests }}>
       Execute the test block</label>
     <button type="submit">Evaluate</button>
   </div>
 </form>

 {% if res %}
 <h2>Smell detectors &mdash; the operational label</h2>
 <div class="card full">
   {% if res.problems %}
     <p class="detfail">&#9888;&nbsp; The smell detectors could not run here, so no labels were produced
     &mdash; this is <em>not</em> the same as &ldquo;clean&rdquo;. The structural measures below are unaffected.</p>
     <ul class="detfail-list">{% for p in res.problems %}<li><code>{{ p }}</code></li>{% endfor %}</ul>
     <p class="hint">Usually a virtual-env problem &mdash; launch with the project venv:
     <code>.venv/Scripts/python dashboard/app.py</code>. If it persists, send me this message.</p>
   {% elif res.smells %}
     {% for s in res.smells %}<span class="chip">{{ s }}</span>{% endfor %}
     {% for d in res.detections %}
       <div class="det">
         <p class="det-h"><b>{{ d.smell }}</b>{% if d.trust %} <span class="rely">rely on {{ d.trust[3] }}</span>{% endif %}</p>
         {% for line, rule, msg in d.locs %}
           <p class="loc"><span class="ln">{% if line %}line {{ line }}{% else %}&mdash;{% endif %}</span>
             <code>{{ rule }}</code> {{ msg }}</p>
         {% endfor %}
       </div>
     {% endfor %}
     {% for s, rows in res.compare %}
       {% if rows %}
       <table class="cmp"><caption>How this snippet sits for <code>{{ s }}</code> &mdash; real medians</caption>
       <tr><th>measure</th><th>clean</th><th>you</th><th>smelly</th></tr>
       {% for name, clean, you, smelly in rows %}
         <tr><td><code>{{ name }}</code></td><td>{{ clean }}</td><td class="you">{{ you }}</td><td>{{ smelly }}</td></tr>
       {% endfor %}
       </table>
       {% endif %}
     {% endfor %}
     {% if res.compare %}<p class="hint" style="margin-top:8px">clean = median of clean code, smelly =
     median of code flagged with that smell, you = this snippet. Shown for the measures that separate the
     smell (real d &ge; 1); a smell with no table here is one structure cannot see &mdash; rely on the
     detector.</p>{% endif %}
   {% else %}
     <span class="clean">No tracked smells detected</span>
     <p class="hint" style="margin-top:10px">The detectors are strict and threshold-based &mdash; they
     flag a smell only when code crosses a specific rule (nesting deeper than 5 levels, 6+ parameters,
     a magic number inside a comparison, a mutable default argument, an unused variable, and so on).
     Code that looks untidy to a human can still pass all twelve. The
     <a href="https://kaiyi03.github.io/Code-Smells/smell_injection/smell_guide.html" target="_blank">smell
     guide</a> shows exactly what each detector looks for.</p>
   {% endif %}
 </div>

 <div class="cards">
   <div class="card">
     <h2 style="margin-top:2px">Structural measures</h2>
     <table><tr><th>measure</th><th>value</th><th>vs clean</th><th>what it is</th></tr>
     {% for name, val, blurb, base in res.structural %}
       <tr><td><code>{{ name }}</code></td><td class="num">{{ val|safe }}</td>
           <td class="base">{% if base %}<span class="{{ base[1] }}">{{ base[0]|safe }}</span>{% else %}<span class="muted">&mdash;</span>{% endif %}</td>
           <td class="blurb">{{ blurb }}</td></tr>
     {% endfor %}
     </table>
     <p class="muted" style="margin-top:8px">&ldquo;vs clean&rdquo; = percentile against 1000 real clean
     functions. Elevated = top 25%, High = top 5% (for maintainability, where lower is worse,
     Low = bottom 25%).</p>
   </div>

   <div class="card">
     <h2 style="margin-top:2px">Similarity vs. reference</h2>
     {% if res.similarity %}
       <table><tr><th>measure</th><th>score</th><th>what it is</th></tr>
       {% for name, val, blurb in res.similarity %}
         <tr><td><code>{{ name }}</code></td><td class="num">{{ val|safe }}</td>
             <td class="blurb">{{ blurb }}</td></tr>
       {% endfor %}
       </table>
       <p class="muted" style="margin-top:8px">100 = identical to the reference; lower = more divergence.</p>
     {% else %}
       <p class="muted">Add a reference solution above to compute similarity
       (BLEU, chrF, ROUGE-L, METEOR, CodeBLEU, AST edit distance).</p>
     {% endif %}
   </div>
 </div>

 <h2>Correctness</h2>
 <div class="card full">
   {% if res.correctness %}
     <span class="badge {{ res.correctness }}">{{ res.correctness }}</span>
     <span class="muted">&nbsp; ran the code against your test block.</span>
   {% elif res.has_tests %}
     <p class="muted">Tests provided &mdash; tick &ldquo;Execute the test block&rdquo; and re-run to check correctness.</p>
   {% else %}
     <p class="muted">Add a test block above (and tick execute) to check whether the code passes.</p>
   {% endif %}
 </div>
 {% endif %}

 <footer>Part of the code-smell benchmarking project &mdash; University of Oxford, Machine Learning Research Group.</footer>
</div></body></html>"""


@app.route("/", methods=["GET", "POST"])
def index():
    code = request.form.get("code", "")
    ref = request.form.get("ref", "")
    tests = request.form.get("tests", "")
    run_tests = bool(request.form.get("run_tests"))
    res = None
    if request.method == "POST" and code.strip():
        res = evaluate(code, ref, tests, run_tests)
    elif request.method == "GET" and not code:      # first load: show a worked example
        code, ref, tests = EXAMPLE_CODE, EXAMPLE_REF, EXAMPLE_TESTS
    return render_template_string(PAGE, code=code, ref=ref, tests=tests,
                                  run_tests=run_tests, res=res)


if __name__ == "__main__":
    # self-check: confirm the subprocess detectors run here, so a broken pylint/ruff
    # is obvious at launch rather than silently reading as "clean code".
    _found, _loc, _probs = detect_labeled("def f(x, acc=[]):\n    return acc\n")
    if _probs or "mutable_default" not in _found:
        print("[WARN] the smell detectors are NOT working here:",
              "; ".join(_probs) or "no smell found on a known-smelly snippet")
        print("       structural measures still work; fix the venv/tools to get smell labels.")
    else:
        print("[ok] smell detectors working")
    if not getattr(BI, "JSCPD", None):
        print("[note] jscpd not found -> duplicate_code will not be detected (npm install -g jscpd)")
    port = int(os.environ.get("PORT", "5000"))
    url = f"http://127.0.0.1:{port}"
    print(f"dashboard ready -> {url}   (keep this window open; Ctrl+C to stop)")
    if not os.environ.get("DASH_NO_BROWSER"):        # open the browser for you
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
