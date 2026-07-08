"""
Walking skeleton for the smell-injection dataset.

Everything runs top to bottom in this one file:
  STEP 1  load clean reference functions (HumanEval + MBPP, or a built-in fallback)
  STEP 2  keep only the ones that are ALREADY clean for our tracked smells
  STEP 3  inject one smell at a time  (automated AST / text transforms)
  STEP 4  re-run the detector to confirm the smell landed and nothing else broke
  STEP 5  save the confirmed (clean, smelly, label) rows to samples.jsonl

All 12 smells are wired up, grouped by the tool that confirms them:
  - pylint : mutable_default, broad_except, unused_import, long_parameter_list,
             deep_nesting, too_many_booleans, long_method, magic_number
  - ruff   : perf_try_in_loop, perf_manual_list, perf_manual_copy
  - jscpd  : duplicate_code
Each is the same shape: write an injector, map it to the rule that proves it landed.

Needs:    pylint, ruff, datasets  (pip)   +   jscpd  (npm install -g jscpd)
Python 3.9+  (uses ast.unparse)
"""

import ast
import copy
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter

# ----------------------------------------------------------------------
# CONFIG  -- the few knobs you'll actually touch
# ----------------------------------------------------------------------
PER_SOURCE = None        # functions per MBPP split / HumanEval; None = all (full build)
TARGET_PER_SMELL = 100   # cap per smell so common smells don't drown the rare ones
# CodeSearchNet top-up (full build only): pull real GitHub Python functions the
# perf injectors can apply to, to lift perf_manual_list/copy toward target.
CSN_COPY_TARGET = 400    # functions with a list-copy shape (the rarer one)
CSN_LIST_TARGET = 200    # functions with a transforming-comprehension shape
CSN_SCAN_LIMIT = 60000   # max functions to stream while hunting for those shapes
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(OUT_DIR, "samples.jsonl")

# smell name -> the pylint message id(s) that mark it.
# broad_except lists two ids so this works on both old and new pylint.
PYLINT_SMELLS = {
    "mutable_default":     ["W0102"],           # def f(x=[])  dangerous-default-value
    "broad_except":        ["W0718", "W0703"],  # except Exception
    "unused_import":       ["W0611"],           # import never used
    "long_parameter_list": ["R0913"],           # too-many-arguments (>5)
    "deep_nesting":        ["R1702"],           # too-many-nested-blocks (>5)
    "too_many_booleans":   ["R0916"],           # too-many-boolean-expressions (>5)
    "long_method":         ["R0915"],           # too-many-statements (>50)
    "magic_number":        ["R2004"],           # magic-value-comparison
}
MSGID_TO_SMELL = {mid: s for s, ids in PYLINT_SMELLS.items() for mid in ids}
PYLINT_ENABLE = ",".join(MSGID_TO_SMELL)

# smell name -> the ruff rule code(s) that mark it (the performance smells)
RUFF_SMELLS = {
    "perf_try_in_loop": ["PERF203"],   # try/except inside a loop
    "perf_manual_list": ["PERF401"],   # manual loop that should be a comprehension
    "perf_manual_copy": ["PERF402"],   # manual loop that just copies a list
}
RULE_TO_SMELL = {code: s for s, codes in RUFF_SMELLS.items() for code in codes}
RUFF_SELECT = ",".join(RULE_TO_SMELL)

# duplicate code is detected by jscpd (a Node tool). A clone must be at least
# this many lines / tokens to count -- our injector duplicates a whole function.
DUP_MIN_LINES = 5
DUP_MIN_TOKENS = 25


def _find_jscpd():
    """Locate jscpd, PATH-independent. Prefer a copy installed locally in this
    folder's node_modules (reproducible, like the .venv); then a JSCPD_PATH
    override; then PATH; then npm's default global dir."""
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, "node_modules", ".bin",
                         "jscpd.cmd" if os.name == "nt" else "jscpd")
    if os.path.exists(local):
        return local
    override = os.environ.get("JSCPD_PATH")
    if override and os.path.exists(override):
        return override
    exe = shutil.which("jscpd")
    if exe:
        return exe
    appdata = os.environ.get("APPDATA", "")
    for name in ("jscpd.cmd", "jscpd.CMD", "jscpd"):
        cand = os.path.join(appdata, "npm", name) if appdata else ""
        if cand and os.path.exists(cand):
            return cand
    return None


JSCPD = _find_jscpd()


# ----------------------------------------------------------------------
# STEP 1 -- load clean reference functions
# ----------------------------------------------------------------------
# Used only if the HuggingFace download is unavailable, so the pipeline
# still runs and produces output today.
FALLBACK = [
    {"id": "fb_1", "source": "fallback",
     "code": "def add(a, b):\n    return a + b\n"},
    {"id": "fb_2", "source": "fallback",
     "code": "def total(items):\n    result = 0\n    for x in items:\n        result += x\n    return result\n"},
    {"id": "fb_3", "source": "fallback",
     "code": "def greet(name):\n    message = 'hello ' + name\n    return message\n"},
    {"id": "fb_4", "source": "fallback",
     "code": "def clip(value, low, high):\n    if value < low:\n        return low\n    if value > high:\n        return high\n    return value\n"},
]


def load_references(per_source=None):
    """Pull clean reference functions. per_source caps how many to take from
    each MBPP split and from HumanEval; None falls back to PER_SOURCE (None = all).
    On a full build it also streams CodeSearchNet for perf-smell source code."""
    refs = []
    full = per_source is None and PER_SOURCE is None
    if per_source is None:
        per_source = PER_SOURCE
    cap = per_source if per_source else 10 ** 9

    def _mbpp():
        from datasets import load_dataset
        for split in ("train", "test", "validation"):
            ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
            for ex in list(ds)[:cap]:
                refs.append({"id": f"mbpp_{ex['task_id']}", "source": "mbpp",
                             "code": ex["code"]})

    def _humaneval():
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        for ex in list(ds)[:cap]:
            # HumanEval gives the signature+docstring (prompt) and the body
            # (canonical_solution) separately; glue them into one function.
            refs.append({"id": ex["task_id"].replace("/", "_"), "source": "humaneval",
                         "code": ex["prompt"] + ex["canonical_solution"]})

    def _codesearchnet():
        # Stream real GitHub Python functions (CodeSearchNet, via CodeXGLUE) and
        # keep only those the perf injectors can apply to. This tops up the two
        # perf smells, which MBPP+HumanEval barely contain. Bucketed so the
        # rarer copy shape is collected in enough numbers.
        from datasets import load_dataset
        ds = load_dataset("google/code_x_glue_ct_code_to_text", "python",
                          split="train", streaming=True)
        copy_n = list_n = scanned = 0
        for ex in ds:
            scanned += 1
            if scanned > CSN_SCAN_LIMIT:
                break
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    norm = ast.unparse(ast.parse(ex.get("code") or ""))
            except (SyntaxError, ValueError, RecursionError):
                continue
            if copy_n < CSN_COPY_TARGET and inject_perf_manual_copy(norm) is not None:
                refs.append({"id": f"csn_{scanned}", "source": "codesearchnet", "code": norm})
                copy_n += 1
            elif list_n < CSN_LIST_TARGET and inject_perf_manual_list(norm) is not None:
                refs.append({"id": f"csn_{scanned}", "source": "codesearchnet", "code": norm})
                list_n += 1
            if copy_n >= CSN_COPY_TARGET and list_n >= CSN_LIST_TARGET:
                break
        print(f"[csn] scanned {scanned}: added {copy_n} copy-shaped + {list_n} list-shaped")

    for loader, label in ((_mbpp, "mbpp"), (_humaneval, "humaneval")):
        try:
            loader()
        except Exception as e:
            print(f"[warn] {label} load failed: {type(e).__name__}: {e}")

    if full:
        try:
            _codesearchnet()
        except Exception as e:
            print(f"[warn] codesearchnet load failed: {type(e).__name__}: {e}")

    if not refs:
        print("=" * 62)
        print("[warn] Could NOT load MBPP/HumanEval from Hugging Face.")
        print(f"       Falling back to {len(FALLBACK)} built-in demo functions.")
        print("=" * 62)
        refs = list(FALLBACK)

    # Normalise via AST round-trip (fixes MBPP \r chars, makes the clean baseline
    # match injector formatting) and drop exact-duplicate reference functions.
    good, seen = [], set()
    for r in refs:
        code = r["code"].replace("\r\n", "\n").replace("\r", "\n")
        try:
            with warnings.catch_warnings():
                # MBPP/HumanEval regex strings like "\d" (not raw) emit a
                # harmless SyntaxWarning; silence it -- the code still parses.
                warnings.simplefilter("ignore", SyntaxWarning)
                norm = ast.unparse(ast.parse(code))
        except (SyntaxError, ValueError):
            continue
        if norm in seen:
            continue
        seen.add(norm)
        r["code"] = norm
        good.append(r)
    return good


# ----------------------------------------------------------------------
# STEP 3 -- the injectors  (this is the "automated" part)
# each takes clean source, returns smelly source (or None if it can't apply)
# ----------------------------------------------------------------------
def _first_func(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def inject_mutable_default(src):
    """Add a parameter with a shared mutable default:  def f(...) -> def f(..., cache=[])"""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    fn.args.args.append(ast.arg(arg="cache"))
    fn.args.defaults.append(ast.List(elts=[], ctx=ast.Load()))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def inject_broad_except(src):
    """Wrap the body in  try: <body> except Exception: return None"""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    body = fn.body
    # keep a leading docstring outside the try, so it still reads normally
    doc, rest = [], body
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        doc, rest = [body[0]], body[1:]
    if not rest:
        return None
    handler = ast.ExceptHandler(
        type=ast.Name(id="Exception", ctx=ast.Load()),
        name=None,
        body=[ast.Return(value=ast.Constant(value=None))],
    )
    fn.body = doc + [ast.Try(body=rest, handlers=[handler], orelse=[], finalbody=[])]
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def inject_unused_import(src):
    """Prepend an import the function never uses."""
    candidates = ["os", "sys", "math", "json", "random", "itertools"]
    used = set(re.findall(r"[A-Za-z_]\w*", src))
    name = next((c for c in candidates if c not in used), None)
    if name is None:
        return None
    return f"import {name}\n" + src


# ---- pylint-confirmed structural injectors --------------------------

def _split_docstring(body):
    """Return (docstring_stmts, rest) so injectors can insert AFTER a docstring."""
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)):
        return [body[0]], body[1:]
    return [], body


def _first_param(fn):
    args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
    return args[0].arg if args else None


def inject_long_parameter_list(src):
    """Append 6 extra keyword params so the signature exceeds the arg limit (>5)."""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    for i in range(6):
        fn.args.args.append(ast.arg(arg=f"extra_{i}"))
        fn.args.defaults.append(ast.Constant(value=None))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def inject_deep_nesting(src):
    """Wrap the body in 6 nested `for` loops so nesting depth exceeds the limit (>5)."""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    doc, rest = _split_docstring(fn.body)
    if not rest:
        return None
    inner = rest
    for i in range(6):
        inner = [ast.For(
            target=ast.Name(id=f"_n{i}", ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id="range", ctx=ast.Load()),
                          args=[ast.Constant(value=1)], keywords=[]),
            body=inner, orelse=[])]
    fn.body = doc + inner
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def inject_too_many_booleans(src):
    """Add an `if` whose condition chains many boolean terms (>5)."""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    p = _first_param(fn)
    if p is None:
        return None
    doc, rest = _split_docstring(fn.body)
    cond = ast.BoolOp(op=ast.And(),
                      values=[ast.Name(id=p, ctx=ast.Load()) for _ in range(8)])
    guard = ast.If(test=cond, body=[ast.Pass()], orelse=[])
    fn.body = doc + [guard] + rest
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def inject_long_method(src):
    """Pad the body with ~55 trivial statements so it exceeds the limit (>50)."""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    doc, rest = _split_docstring(fn.body)
    pad = [ast.Assign(targets=[ast.Name(id="_acc", ctx=ast.Store())],
                      value=ast.Constant(value=0))]
    for i in range(1, 56):
        pad.append(ast.Assign(
            targets=[ast.Name(id="_acc", ctx=ast.Store())],
            value=ast.BinOp(left=ast.Name(id="_acc", ctx=ast.Load()),
                            op=ast.Add(), right=ast.Constant(value=i))))
    fn.body = doc + pad + rest
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def inject_magic_number(src):
    """Add a comparison against a bare literal (a 'magic number')."""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    p = _first_param(fn)
    doc, rest = _split_docstring(fn.body)
    left = (ast.Name(id=p, ctx=ast.Load()) if p is not None
            else ast.Call(func=ast.Name(id="len", ctx=ast.Load()),
                          args=[ast.Constant(value="x")], keywords=[]))
    cmp = ast.Compare(left=left, ops=[ast.Eq()],
                      comparators=[ast.Constant(value=42)])
    stmt = ast.Assign(targets=[ast.Name(id="_", ctx=ast.Store())], value=cmp)
    fn.body = doc + [stmt] + rest
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ---- performance-smell injectors (confirmed by ruff) ----------------
# These DE-OPTIMISE an existing shape (loop / comprehension / copy) rather
# than adding new code, so the touched variable stays used and no second
# smell sneaks in. They apply only where the shape exists -> lower yield.

def inject_perf_try_in_loop(src):
    """Wrap an existing loop's body in try/except ValueError (PERF203).
    ValueError, not Exception, so it isn't ALSO a broad_except."""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    for node in ast.walk(fn):
        if isinstance(node, (ast.For, ast.While)) and node.body:
            handler = ast.ExceptHandler(type=ast.Name(id="ValueError", ctx=ast.Load()),
                                        name=None, body=[ast.Pass()])
            node.body = [ast.Try(body=node.body, handlers=[handler], orelse=[], finalbody=[])]
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
    return None


def _simple_listcomp(stmt):
    """True if stmt is `name = [elt for t in it]` with one plain generator."""
    return (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.ListComp)
            and len(stmt.value.generators) == 1
            and not stmt.value.generators[0].ifs
            and not stmt.value.generators[0].is_async)


def _is_identity(comp):
    """True if the comprehension is `[x for x in it]` (a copy, not a transform)."""
    g = comp.generators[0]
    return (isinstance(comp.elt, ast.Name) and isinstance(g.target, ast.Name)
            and comp.elt.id == g.target.id)


def _append_loop(name, target, it, value):
    """Build:  name = []   /   for <target> in <it>: name.append(<value>)"""
    init = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                      value=ast.List(elts=[], ctx=ast.Load()))
    loop = ast.For(target=target, iter=it, body=[ast.Expr(ast.Call(
        func=ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()),
                           attr="append", ctx=ast.Load()),
        args=[value], keywords=[]))], orelse=[])
    return [init, loop]


def inject_perf_manual_list(src):
    """Turn a transforming list comprehension into a manual append loop (PERF401)."""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    for i, stmt in enumerate(fn.body):
        if _simple_listcomp(stmt) and not _is_identity(stmt.value):
            comp = stmt.value
            g = comp.generators[0]
            fn.body[i:i + 1] = _append_loop(stmt.targets[0].id, g.target, g.iter, comp.elt)
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
    return None


def _provably_list(fn, node):
    """True only if `node` is guaranteed to be a list, so rewriting a copy of it
    as a []-append loop preserves behaviour. list(...) / [..] / [x for x in ..]
    are always lists; a bare name counts only if it was assigned one of those
    earlier in the function. Anything unprovable (e.g. a plain parameter, which
    could be a dict or set) is treated as NOT a list -- this is what stops us
    turning `d = some_dict.copy()` into a broken list-append loop."""
    def direct(n):
        return (isinstance(n, (ast.List, ast.ListComp))
                or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "list"))
    if direct(node):
        return True
    if isinstance(node, ast.Name):
        return any(isinstance(s, ast.Assign) and len(s.targets) == 1
                   and isinstance(s.targets[0], ast.Name)
                   and s.targets[0].id == node.id and direct(s.value)
                   for s in fn.body)
    return False


def inject_perf_manual_copy(src):
    """Turn a *list* copy into an append loop (PERF402). Handles `[x for x in it]`
    and `list(it)` (always lists), plus `it[:]` and `it.copy()` ONLY when the
    source is provably a list -- the guard stops dict/set copies, whose loop
    rewrite would silently change behaviour, from being injected."""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    for i, stmt in enumerate(fn.body):
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)):
            continue
        name, val = stmt.targets[0].id, stmt.value
        target = it = None
        if _simple_listcomp(stmt) and _is_identity(val):                     # [x for x in it]
            target, it = val.generators[0].target, val.generators[0].iter
        elif (isinstance(val, ast.Call) and isinstance(val.func, ast.Name)   # list(it)
              and val.func.id == "list" and len(val.args) == 1 and not val.keywords):
            target, it = ast.Name(id="_c", ctx=ast.Store()), val.args[0]
        elif (isinstance(val, ast.Subscript) and isinstance(val.slice, ast.Slice)  # it[:]
              and val.slice.lower is None and val.slice.upper is None and val.slice.step is None
              and _provably_list(fn, val.value)):
            target, it = ast.Name(id="_c", ctx=ast.Store()), val.value
        elif (isinstance(val, ast.Call) and isinstance(val.func, ast.Attribute)     # it.copy()
              and val.func.attr == "copy" and not val.args and not val.keywords
              and _provably_list(fn, val.func.value)):
            target, it = ast.Name(id="_c", ctx=ast.Store()), val.func.value
        if it is None:
            continue
        fn.body[i:i + 1] = _append_loop(name, target, it, ast.Name(id=target.id, ctx=ast.Load()))
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    return None


# ---- duplicate-code injector (confirmed by jscpd) -------------------

def inject_duplicate_code(src):
    """Append a renamed copy of the function, creating a duplicated block."""
    tree = ast.parse(src)
    fn = _first_func(tree)
    if fn is None:
        return None
    dup = copy.deepcopy(fn)
    dup.name = fn.name + "_copy"
    tree.body.append(dup)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


INJECTORS = {
    "mutable_default": inject_mutable_default,
    "broad_except": inject_broad_except,
    "unused_import": inject_unused_import,
    "long_parameter_list": inject_long_parameter_list,
    "deep_nesting": inject_deep_nesting,
    "too_many_booleans": inject_too_many_booleans,
    "long_method": inject_long_method,
    "magic_number": inject_magic_number,
    "perf_try_in_loop": inject_perf_try_in_loop,
    "perf_manual_list": inject_perf_manual_list,
    "perf_manual_copy": inject_perf_manual_copy,
    "duplicate_code": inject_duplicate_code,
}


# ----------------------------------------------------------------------
# STEP 4 -- the detector: which tracked smells does this code contain?
# Runs pylint (for the 8 pylint smells) AND ruff (for the 3 perf smells),
# each restricted to ONLY our tracked rules, then unions the results. So
# "collateral" is measured against exactly the smells we track.
# ----------------------------------------------------------------------
def _pylint_smells(path):
    proc = subprocess.run(
        [sys.executable, "-m", "pylint", path,
         "--load-plugins=pylint.extensions.magic_value",  # needed for R2004
         "--output-format=json", "--disable=all", f"--enable={PYLINT_ENABLE}"],
        capture_output=True, text=True,
    )
    try:
        messages = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        messages = []
    return {MSGID_TO_SMELL[m["message-id"]] for m in messages
            if m.get("message-id") in MSGID_TO_SMELL}


def _ruff_smells(path):
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", path,
         "--isolated", "--select", RUFF_SELECT, "--output-format=json"],
        capture_output=True, text=True,
    )
    try:
        messages = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        messages = []
    return {RULE_TO_SMELL[m["code"]] for m in messages
            if m.get("code") in RULE_TO_SMELL}


def detect(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        path = f.name
    try:
        return _pylint_smells(path) | _ruff_smells(path)
    finally:
        os.unlink(path)


def has_duplicate(code):
    """True if jscpd finds a duplicated block within `code`."""
    if not JSCPD:
        return False
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "snippet.py")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)
        out = os.path.join(tmp, "report")
        cmd = [JSCPD, src, "--min-lines", str(DUP_MIN_LINES),
               "--min-tokens", str(DUP_MIN_TOKENS),
               "--reporters", "json", "--output", out, "--silent"]
        if os.name == "nt":
            cmd = ["cmd", "/c", *cmd]
        subprocess.run(cmd, capture_output=True, text=True)
        report = os.path.join(out, "jscpd-report.json")
        if not os.path.exists(report):
            return False
        with open(report, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("statistics", {}).get("total", {}).get("clones", 0) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def all_smells(code, check_dup=False):
    """All tracked smells in `code`. jscpd (duplicate_code) runs only when
    check_dup is set -- it's slow and no other injector can create it."""
    found = detect(code)
    if check_dup and has_duplicate(code):
        found.add("duplicate_code")
    return found


def detect_many(items):
    """Batched pylint+ruff. items: {key: code}. Returns {key: set(smells)}.
    Runs each tool ONCE over a temp folder of all snippets instead of once per
    snippet -- the big speed-up for the full build. (jscpd is per-file and is
    handled separately, only for duplicate_code.)"""
    if not items:
        return {}
    tmp = tempfile.mkdtemp()
    fname_to_key = {}
    try:
        for i, (key, code) in enumerate(items.items()):
            fname = f"s{i}.py"
            fname_to_key[fname] = key
            with open(os.path.join(tmp, fname), "w", encoding="utf-8") as f:
                f.write(code)
        result = {key: set() for key in items}

        pyl = subprocess.run(
            [sys.executable, "-m", "pylint", "--recursive=y", tmp,
             "--load-plugins=pylint.extensions.magic_value",
             "--output-format=json", "--disable=all", f"--enable={PYLINT_ENABLE}",
             "--jobs=0"],
            capture_output=True, text=True)
        try:
            for m in json.loads(pyl.stdout or "[]"):
                key = fname_to_key.get(os.path.basename(m.get("path") or m.get("abspath") or ""))
                smell = MSGID_TO_SMELL.get(m.get("message-id"))
                if key and smell:
                    result[key].add(smell)
        except json.JSONDecodeError:
            pass

        ruf = subprocess.run(
            [sys.executable, "-m", "ruff", "check", tmp,
             "--isolated", "--select", RUFF_SELECT, "--output-format=json"],
            capture_output=True, text=True)
        try:
            for m in json.loads(ruf.stdout or "[]"):
                key = fname_to_key.get(os.path.basename(m.get("filename") or ""))
                smell = RULE_TO_SMELL.get(m.get("code"))
                if key and smell:
                    result[key].add(smell)
        except json.JSONDecodeError:
            pass
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------
# STEP 2 + 5 -- filter to clean, inject, verify, save
# ----------------------------------------------------------------------
def build():
    t0 = time.time()
    refs = load_references()
    print(f"loaded {len(refs)} reference functions")
    if not JSCPD:
        print("[note] jscpd not found -- duplicate_code will yield 0.")

    # STEP 2 -- clean filter, batched. jscpd is skipped here: a lone reference
    # function almost never contains internal duplication, and per-file jscpd
    # over every reference would dominate the runtime.
    base = detect_many({r["id"]: r["code"] for r in refs})
    clean = [r for r in refs if not base.get(r["id"])]
    # Shuffle MBPP+HumanEval together (so the caps draw a mix of the two), but
    # keep CodeSearchNet AFTER them: the 10 common smells then fill entirely from
    # MBPP/HumanEval, and CodeSearchNet only tops up the perf smells still under
    # target. Both groups shuffled with a fixed seed for reproducibility.
    rng = random.Random(0)
    noncsn = [r for r in clean if r["source"] != "codesearchnet"]
    csn = [r for r in clean if r["source"] == "codesearchnet"]
    rng.shuffle(noncsn)
    rng.shuffle(csn)
    clean = noncsn + csn
    clean_by_id = {r["id"]: r["code"] for r in clean}
    print(f"{len(clean)} clean references -> injecting (target {TARGET_PER_SMELL}/smell)\n")

    records, kept, dropped = [], Counter(), Counter()
    CHUNK = 150
    for start in range(0, len(clean), CHUNK):
        chunk = clean[start:start + CHUNK]
        # STEP 3 -- generate candidates, skipping smells already at target
        cand = []   # list of (ref_id, smell, smelly_code)
        for r in chunk:
            for smell, inject in INJECTORS.items():
                if kept[smell] >= TARGET_PER_SMELL:
                    continue
                try:
                    smelly = inject(r["code"])
                except Exception:
                    smelly = None
                if smelly:
                    cand.append((r["id"], smell, smelly))
                else:
                    dropped[f"{smell} (injector could not apply)"] += 1
        if not cand:
            continue
        # STEP 4 -- verify the whole chunk in ONE batched pylint+ruff pass
        verdict = detect_many({str(i): code for i, (_, _, code) in enumerate(cand)})
        for i, (rid, smell, code) in enumerate(cand):
            if kept[smell] >= TARGET_PER_SMELL:
                continue
            found = set(verdict.get(str(i), set()))
            if smell == "duplicate_code" and has_duplicate(code):   # per-file jscpd
                found.add("duplicate_code")
            collateral = sorted(found - {smell})
            if smell in found and not collateral:
                records.append({
                    "id": f"{rid}__{smell}", "source_task": rid, "smell": smell,
                    "clean_code": clean_by_id[rid], "smelly_code": code,
                    "detector_confirmed": True, "collateral": [],
                })
                kept[smell] += 1
            elif smell not in found:
                dropped[f"{smell} (smell not detected)"] += 1
            else:
                dropped[f"{smell} (collateral {collateral})"] += 1
        if all(kept[s] >= TARGET_PER_SMELL for s in INJECTORS):
            break

    # the clean versions are the negatives ("no smell"), capped at the target too
    for r in clean[:TARGET_PER_SMELL]:
        records.append({
            "id": f"{r['id']}__clean", "source_task": r["id"], "smell": "clean",
            "clean_code": r["code"], "smelly_code": None,
            "detector_confirmed": True, "collateral": [],
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:           # STEP 5
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print("kept (confirmed smelly samples):")
    for smell in INJECTORS:
        flag = "" if kept[smell] >= TARGET_PER_SMELL else "   << under target"
        print(f"  {smell:22} {kept[smell]}{flag}")
    print(f"  {'clean negatives':22} {sum(1 for r in records if r['smell'] == 'clean')}")
    if dropped:
        print("\ndropped (top reasons):")
        for reason, n in dropped.most_common(10):
            print(f"  {n:5}  {reason}")
    print(f"\nwrote {len(records)} rows to {OUT_FILE}  (elapsed {time.time() - t0:.0f}s)")


def _bootstrap():
    """If this project's .venv exists and we're not already using it, re-launch
    with it. This makes `python build_injected.py` work no matter which
    interpreter you start from (global Python, VSCode Run button, etc.)."""
    here = os.path.dirname(os.path.abspath(__file__))
    venv_py = os.path.abspath(os.path.join(here, os.pardir, ".venv", "Scripts", "python.exe"))
    if os.path.exists(venv_py) and os.path.abspath(sys.executable).lower() != venv_py.lower():
        print(f"[setup] switching to project venv:\n        {venv_py}\n")
        raise SystemExit(subprocess.run([venv_py, os.path.abspath(__file__), *sys.argv[1:]]).returncode)


def _preflight():
    """Verify each detector actually works here before building, so we never
    write a partial/confusing file. Prints one status line per tool."""
    checks = [
        ("pylint core",        "def f(x=[]):\n    return x\n",                          "mutable_default"),
        ("pylint magic_value", "def f(x):\n    _ = x == 42\n    return x\n",             "magic_number"),
        ("ruff (perf)",        "def f(xs):\n    o = []\n    for x in xs:\n        o.append(x + 1)\n    return o\n", "perf_manual_list"),
    ]
    ok = True
    for label, code, smell in checks:
        working = smell in detect(code)
        ok = ok and working
        print(f"[preflight] {label:20} {'OK' if working else 'NOT WORKING'}")
    print(f"[preflight] {'jscpd (duplicate)':20} " +
          (f"OK  ({JSCPD})" if JSCPD else "NOT FOUND -> duplicate_code will be 0; run: npm install -g jscpd"))
    if not ok:
        print("\n[preflight] a core detector is broken -- stopping. Install tools into this interpreter:")
        print(f'    "{sys.executable}" -m pip install pylint ruff datasets')
        raise SystemExit(1)
    print()


if __name__ == "__main__":
    _bootstrap()    # make sure we're on the venv that has the packages
    _preflight()    # make sure the detector actually works before building
    build()
