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

import os
import subprocess
import sys

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
from build_injected import all_smells                       # noqa: E402
from correctness import run_program                         # noqa: E402

try:
    from detection_report import TRUST                      # noqa: E402
except Exception:
    TRUST = []
TRUST_MAP = {row[0]: row for row in TRUST}   # smell -> (smell, structural, similarity, use)

STRUCT = [m for m in PANEL if not m.needs_ref]
SIM = [m for m in PANEL if m.needs_ref]

app = Flask(__name__)


def _fmt(v, dp):
    return "&mdash;" if v is None else f"{v:.{dp}f}"


def evaluate(code, ref, tests, run_tests):
    """Run the panel on one snippet. Returns a dict of display-ready pieces."""
    smells = sorted(all_smells(code, check_dup=True))
    structural = [(m.name, _fmt(m.fn(code), 2), m.blurb) for m in STRUCT]

    similarity = None
    if ref.strip():
        similarity = [(m.name, _fmt(m.fn(code, ref), 1), m.blurb) for m in SIM]

    correctness = None
    if tests.strip() and run_tests:
        correctness = run_program(code + "\n\n" + tests)

    hints = [(s, TRUST_MAP.get(s)) for s in smells]
    return {"smells": smells, "hints": hints, "structural": structural,
            "similarity": similarity, "correctness": correctness,
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
 .hint b{color:#1a1a1a}
 .pass{background:#dcfce7;color:#15803d} .fail{background:#fee2e2;color:#b91c1c}
 .timeout{background:#fef9c3;color:#a16207}
 .badge{display:inline-block;border-radius:999px;padding:4px 14px;font-size:13px;font-weight:600}
 .muted{color:#999;font-size:12.5px}
 .warn{color:#b45309;font-size:11.5px;margin:4px 0 0}
 footer{margin-top:36px;font-size:12.5px;color:#999}
 @media (max-width:720px){.two,.cards{grid-template-columns:1fr}}
 @media (prefers-color-scheme:dark){
   body{background:#0f1115;color:#e6e6e6}.sub,.muted{color:#8a8f98}
   form,.card{background:#171a21;border-color:#2a2f3a;box-shadow:none}
   h2{color:#a8adb7;border-color:#2a2f3a}label{color:#c3c8d2}
   textarea{background:#0f1115;border-color:#2a2f3a;color:#e6e6e6}
   th,td{border-color:#242a33}td.blurb,th{color:#7a808a}
   code{background:#222732}.hint{color:#a8adb7}.hint b{color:#e6e6e6}
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
   {% if res.smells %}
     {% for s in res.smells %}<span class="chip">{{ s }}</span>{% endfor %}
     {% for s, t in res.hints %}
       {% if t %}<p class="hint"><b>{{ s }}</b> &mdash; structural: {{ t[1] }}, similarity: {{ t[2] }}.
         Rely on: {{ t[3] }}.</p>{% endif %}
     {% endfor %}
   {% else %}
     <span class="clean">No tracked smells detected</span>
   {% endif %}
 </div>

 <div class="cards">
   <div class="card">
     <h2 style="margin-top:2px">Structural measures</h2>
     <table><tr><th>measure</th><th>value</th><th>what it is</th></tr>
     {% for name, val, blurb in res.structural %}
       <tr><td><code>{{ name }}</code></td><td class="num">{{ val|safe }}</td>
           <td class="blurb">{{ blurb }}</td></tr>
     {% endfor %}
     </table>
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
    return render_template_string(PAGE, code=code, ref=ref, tests=tests,
                                  run_tests=run_tests, res=res)


if __name__ == "__main__":
    print("dashboard on http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
