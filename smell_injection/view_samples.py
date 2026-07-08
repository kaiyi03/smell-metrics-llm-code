"""
Human-friendly view of samples.jsonl.

Prints a summary to the terminal, then writes samples_report.html -- open that
in your browser to see each clean-vs-smelly pair side by side, with the changes
highlighted (like a GitHub / VSCode diff). You do NOT need to read the raw
.jsonl by hand.

Uses only the Python standard library, so it runs with ANY Python:
    python view_samples.py
"""

import json
import os
import html
import difflib
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples.jsonl")
REPORT = os.path.join(HERE, "samples_report.html")


def load():
    if not os.path.exists(SAMPLES):
        raise SystemExit(f"no samples file at {SAMPLES} -- run build_injected.py first")
    with open(SAMPLES, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_of(rec):
    t = rec["source_task"]
    if t.startswith("mbpp"):
        return "mbpp"
    if t.startswith("HumanEval"):
        return "humaneval"
    if t.startswith("csn"):
        return "codesearchnet"
    if t.startswith("fb"):
        return "fallback (datasets did NOT load!)"
    return "other"


def print_summary(rows):
    by_smell = Counter(r["smell"] for r in rows)
    by_source = Counter(source_of(r) for r in rows)
    print(f"total rows: {len(rows)}\n")
    print("by label:")
    for k, v in by_smell.most_common():
        print(f"  {k:20} {v}")
    print("\nby source:")
    for k, v in by_source.most_common():
        print(f"  {k:20} {v}")
    return by_smell


CSS = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1c1c1c; }
h1 { font-size: 20px; }
h2 { font-size: 16px; margin-top: 30px; border-bottom: 2px solid #ddd; padding-bottom: 4px; }
.summary { border-collapse: collapse; margin: 10px 0 4px; }
.summary td { padding: 3px 16px 3px 0; }
.card { border: 1px solid #ddd; border-radius: 6px; margin: 14px 0; overflow-x: auto; }
.card .head { background: #f4f4f4; padding: 6px 10px; font-family: Consolas, monospace; font-size: 13px; }
table.diff { width: 100%; border-collapse: collapse; font-family: Consolas, monospace; font-size: 12.5px; }
table.diff td { padding: 0 6px; vertical-align: top; }
.diff_header { color: #aaa; text-align: right; }
.diff_next { background: #f7f7f7; }
.diff_add { background: #d6ffd6; }
.diff_chg { background: #fff4c2; }
.diff_sub { background: #ffd6d6; }
.legend span { padding: 2px 8px; border-radius: 3px; margin-right: 8px; font-size: 12px; }
"""


def write_report(rows):
    smelly = [r for r in rows if r["smell"] != "clean" and r["smelly_code"]]
    hd = difflib.HtmlDiff(wrapcolumn=72)
    by_smell = Counter(r["smell"] for r in rows)

    parts = [f"<style>{CSS}</style>", "<h1>Injected smell samples</h1>"]
    parts.append("<p class='legend'>"
                 "<span class='diff_add'>added</span>"
                 "<span class='diff_chg'>changed</span>"
                 "<span class='diff_sub'>removed</span>"
                 "left = clean, right = smelly</p>")

    parts.append("<table class='summary'>")
    parts.append(f"<tr><td><b>total rows</b></td><td>{len(rows)}</td></tr>")
    for k, v in by_smell.most_common():
        parts.append(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>")
    parts.append("</table>")

    for smell in sorted(set(r["smell"] for r in smelly)):
        group = [r for r in smelly if r["smell"] == smell]
        parts.append(f"<h2>{html.escape(smell)} &mdash; {len(group)} samples</h2>")
        for r in group:
            table = hd.make_table(
                r["clean_code"].splitlines(), r["smelly_code"].splitlines(),
                "clean", "smelly", context=True, numlines=2,
            )
            parts.append(f"<div class='card'><div class='head'>{html.escape(r['id'])}</div>{table}</div>")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"\nwrote report: {REPORT}")
    print("open it in your browser (double-click the file, or drag it into a browser tab).")


if __name__ == "__main__":
    rows = load()
    print_summary(rows)
    write_report(rows)
