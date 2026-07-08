"""
Render generations.jsonl as a readable HTML page: for each task, Qwen's output
side by side with the canonical reference. Much easier to skim than the raw JSON.

    python arc_qwen/view_generations.py      # writes generations_report.html
"""

import html
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "generations.jsonl")
OUT = os.path.join(HERE, "generations_report.html")


def main():
    rows = [json.loads(line) for line in open(IN, encoding="utf-8")]
    src = Counter(r["source"] for r in rows)
    toks = [r["n_output_tokens"] for r in rows]

    cards = []
    for r in rows:
        cards.append(
            f'<div class="card" data-id="{html.escape(r["task_id"].lower())}" '
            f'data-src="{html.escape(r["source"])}">'
            f'<div class="hd"><b>{html.escape(r["task_id"])}</b>'
            f'<span class="tag">{html.escape(r["source"])}</span>'
            f'<span class="muted">{r["n_output_tokens"]} output tokens</span></div>'
            f'<div class="cols">'
            f'<div class="col"><div class="lbl">Qwen output</div>'
            f'<pre>{html.escape(r["generated_code"])}</pre></div>'
            f'<div class="col"><div class="lbl">Reference (canonical)</div>'
            f'<pre>{html.escape(r["canonical_code"])}</pre></div>'
            f'</div></div>'
        )

    summary = " · ".join(f"{k}: {v}" for k, v in src.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Qwen generations</title>
<style>
  body {{ font: 14px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; color: #111;
         max-width: 1150px; margin: 24px auto; padding: 0 16px; }}
  h1 {{ font-size: 22px; margin-bottom: 2px; }}
  .muted {{ color: #666; }}
  .bar {{ position: sticky; top: 0; background: #fff; padding: 10px 0; border-bottom: 1px solid #e1e4e8; }}
  #q {{ width: 280px; padding: 6px 10px; font-size: 14px; border: 1px solid #ccc; border-radius: 6px; }}
  .card {{ border: 1px solid #e1e4e8; border-radius: 8px; margin: 14px 0; overflow: hidden; }}
  .hd {{ background: #f6f8fa; padding: 8px 12px; border-bottom: 1px solid #e1e4e8;
         display: flex; gap: 10px; align-items: center; }}
  .tag {{ font-size: 11px; background: #dbeafe; color: #1e40af; padding: 1px 8px; border-radius: 10px; }}
  .cols {{ display: flex; flex-wrap: wrap; }}
  .col {{ flex: 1 1 380px; min-width: 0; border-right: 1px solid #eee; }}
  .col:last-child {{ border-right: none; }}
  .lbl {{ font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #888;
          padding: 6px 12px 0; }}
  pre {{ margin: 4px 0 0; padding: 10px 12px; overflow-x: auto; font: 12.5px/1.45
         "SF Mono", Consolas, "Liberation Mono", monospace; background: #fbfcfd; }}
</style></head><body>
<h1>Qwen generations</h1>
<p class="muted">{len(rows)} tasks &middot; {summary} &middot; mean {sum(toks) // len(toks)}
   output tokens (median {sorted(toks)[len(toks)//2]})</p>
<div class="bar"><input id="q" placeholder="filter by task id (e.g. mbpp_15, HumanEval_0)"
   oninput="filt()"> <span id="n" class="muted"></span></div>
{"".join(cards)}
<script>
  const cards = [...document.querySelectorAll('.card')];
  const n = document.getElementById('n');
  function filt() {{
    const q = document.getElementById('q').value.trim().toLowerCase();
    let shown = 0;
    for (const c of cards) {{
      const ok = !q || c.dataset.id.includes(q) || c.dataset.src.includes(q);
      c.style.display = ok ? '' : 'none';
      if (ok) shown++;
    }}
    n.textContent = shown + ' shown';
  }}
  filt();
</script>
</body></html>"""

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"wrote {OUT}  ({len(rows)} generations)")


if __name__ == "__main__":
    main()
