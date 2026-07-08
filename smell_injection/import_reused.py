"""
Import the reused code-smell datasets into one normalised file: reused.jsonl.

Sources:
  - codesmelldata2 : CodeSmellExt/eval_dataset.json  -- real functions, Pylint labels
  - codesmelldata1 : CodeSmells_v1/.../codesmell_dataset.csv -- real functions, Pylint labels
  - pysmell        : pythoncodesmell -- HUMAN-labelled Long Method / Long Parameter List;
                     the code is NOT stored, so it is recovered from GitHub at the exact
                     commit + line number.

Each output record:
  id, source, smell, label ("yes"/"no"), code, label_origin ("pylint"/"human"), meta

label_origin is the important one: only "human" labels can test Pylint WITHOUT
circularity (you cannot grade Pylint with labels Pylint produced).

Run with the project venv (needs internet for the pysmell recovery):
    python smell_injection/import_reused.py
"""

import ast
import csv
import json
import os
import urllib.request
from collections import Counter

BASE = r"C:\KY_D\KY 2025 - 2026\Summer 26 Research\_datasets_eval"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reused.jsonl")
MAX_PER = 500              # cap per smell for the Pylint-labelled sources
MAX_PYSMELL_PER = 250      # cap per (smell, label) to bound GitHub downloads

# Pylint message id -> our smell name (same map as build_injected.py)
MSGID_TO_SMELL = {
    "W0102": "mutable_default",
    "W0718": "broad_except", "W0703": "broad_except",
    "W0611": "unused_import",
    "R0913": "long_parameter_list",
    "R1702": "deep_nesting",
    "R0916": "too_many_booleans",
    "R0915": "long_method",
    "R2004": "magic_number",
}

records = []


def add(source, smell, label, code, label_origin, meta):
    if code and code.strip():
        records.append({
            "id": f"{source}_{len(records)}", "source": source, "smell": smell,
            "label": label, "code": code, "label_origin": label_origin, "meta": meta,
        })


# ---------------------------------------------------------------- CodeSmellData 2.0
def import_v2():
    p = os.path.join(BASE, "CodeSmellExt", "dataset", "extracted", "eval_dataset.json")
    d = json.load(open(p, encoding="utf-8"))
    c = Counter()
    for k in d["code"]:
        smell = MSGID_TO_SMELL.get(d["s_msg_id"][k])
        if not smell or c[smell] >= MAX_PER:
            continue
        add("codesmelldata2", smell, "yes", d["code"][k], "pylint",
            {"msg_id": d["s_msg_id"][k], "repo": d["repo"].get(k), "fun": d["fun_name"].get(k)})
        c[smell] += 1
    print("codesmelldata2:", dict(c))


# ---------------------------------------------------------------- CodeSmellData 1.0
def import_v1():
    p = os.path.join(BASE, "CodeSmells_v1", "dataset", "codesmell_dataset.csv")
    csv.field_size_limit(10 ** 9)
    c = Counter()
    with open(p, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            smell = MSGID_TO_SMELL.get(row.get("msg_id"))
            if not smell or c[smell] >= MAX_PER:
                continue
            add("codesmelldata1", smell, "yes", row.get("code"), "pylint",
                {"msg_id": row.get("msg_id"), "repo": row.get("repo"), "fun": row.get("func_name")})
            c[smell] += 1
    print("codesmelldata1:", dict(c))


# ---------------------------------------------------------------- PySmell (human labels)
BS = chr(92)
REPO = {"ansible": "ansible/ansible", "boto": "boto/boto", "django": "django/django",
        "ipython": "ipython/ipython", "matplotlib": "matplotlib/matplotlib", "nltk": "nltk/nltk",
        "numpy": "numpy/numpy", "scipy": "scipy/scipy", "tornado": "tornadoweb/tornado"}
_cache = {}


def _fetch(repo, h, rel):
    url = f"https://raw.githubusercontent.com/{repo}/{h}/{rel}"
    if url not in _cache:
        try:
            with urllib.request.urlopen(url, timeout=30) as f:
                _cache[url] = f.read().decode("utf-8", "replace")
        except Exception:
            _cache[url] = None
    return _cache[url]


def _slice_func(src, line):
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if abs(node.lineno - line) <= 3 and (best is None or abs(node.lineno - line) < abs(best.lineno - line)):
                best = node
    if best is None:
        return None
    return "\n".join(src.splitlines()[best.lineno - 1: best.end_lineno])


def _inrepo_path(winpath):
    parts = winpath.split(BS)
    i = parts.index("9 project")
    return parts[i + 1].lower(), "/".join(parts[i + 3:])


def import_pysmell():
    pcs = os.path.join(BASE, "pythoncodesmell")
    hashes = {}
    for r in csv.DictReader(open(os.path.join(pcs, "metrics from static analysis tools", "project versions.csv"),
                                 encoding="utf-8", errors="replace")):
        if r["Note"].strip() == "9 projects":
            hashes[r["Project"].strip().lower()] = r["Hash"].strip()

    files = {"LongMethod_YesNo.csv": "long_method",
             "LongParameterList_YesNo.csv": "long_parameter_list"}
    c, avail = Counter(), Counter()
    for fname, smell in files.items():
        rows = list(csv.DictReader(open(os.path.join(pcs, "dataset label by chen add new metrics by me", fname),
                                        encoding="utf-8", errors="replace")))
        for r in rows:
            if "9 project" + BS not in r["file"]:
                continue
            label = r["smell"].strip().lower()
            if label not in ("yes", "no"):
                continue
            avail[(smell, label)] += 1
            if c[(smell, label)] >= MAX_PYSMELL_PER:
                continue
            try:
                proj, rel = _inrepo_path(r["file"])
            except ValueError:
                continue
            if proj not in REPO or proj not in hashes:
                continue
            src = _fetch(REPO[proj], hashes[proj], rel)
            if not src:
                continue
            code = _slice_func(src, int(float(r["lineno"])))
            if not code:
                continue
            add("pysmell", smell, label, code, "human",
                {"project": proj, "understandname": r.get("understandname"), "lineno": r.get("lineno")})
            c[(smell, label)] += 1
    print("pysmell recovered:", dict(c))
    print("pysmell available (9-project rows):", dict(avail))


def dedup(recs):
    """Keep one record per (normalised code, smell). A function that carries two
    different smells is kept for each (both are valid test cases); the same
    function+smell is kept once, preferring human labels (pysmell), then v2 over v1."""
    rank = {"pysmell": 0, "codesmelldata2": 1, "codesmelldata1": 2}

    def norm(c):
        return "\n".join(line.rstrip() for line in c.strip().splitlines())

    best = {}
    for r in recs:
        key = (norm(r["code"]), r["smell"])
        cur = best.get(key)
        if cur is None or rank.get(r["source"], 9) < rank.get(cur["source"], 9):
            best[key] = r
    return list(best.values())


if __name__ == "__main__":
    import_v2()
    import_v1()
    import_pysmell()
    clean = dedup(records)
    print(f"\nde-dup: {len(records)} -> {len(clean)} ({len(records) - len(clean)} duplicates removed)")
    with open(OUT, "w", encoding="utf-8") as f:
        for rec in clean:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {len(clean)} reused records to {OUT}")
