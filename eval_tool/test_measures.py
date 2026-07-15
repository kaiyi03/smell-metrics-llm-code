"""Correctness checks for the panel measures -- especially the four added for Noor's
suggestions: cognitive complexity, comment density, function/API usage, AST similarity.

Every expected value is worked out BY HAND from the measure's definition (not read
back from the code), so a passing run means the measure computes what it claims to,
not merely that it is self-consistent.

Run:  .venv/Scripts/python eval_tool/test_measures.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from measures import PANEL   # noqa: E402

M = {m.name: m.fn for m in PANEL}
CHECKS = []


def check(desc, got, expected):
    CHECKS.append((got == expected, desc, got, expected))


def cond(desc, ok, detail):
    CHECKS.append((bool(ok), desc, detail, "true"))


# ---------------------------------------------------------------- cognitive complexity
# SonarSource rule: +1 for each break in linear flow, PLUS the current nesting depth.
FLAT = "def f(x):\n    y = x + 1\n    return y\n"
ONE_IF = "def f(x):\n    if x:\n        return 1\n    return 0\n"
NESTED = ("def f(x):\n    for i in range(x):\n        if i > 2:\n"
          "            if i < 9:\n                return i\n    return 0\n")
check("cognitive: flat function = 0", M["cognitive"](FLAT), 0.0)
check("cognitive: single if = 1", M["cognitive"](ONE_IF), 1.0)
check("cognitive: for(+1) if(+2) if(+3) = 6", M["cognitive"](NESTED), 6.0)

# ---------------------------------------------------------------- function / API usage
# distinct callee names. g, g, xs.append, h  ->  {g, append, h} = 3
CALLS = "def f(xs):\n    a = g(1)\n    b = g(2)\n    xs.append(a)\n    return h(b)\n"
check("api_calls: distinct callees {g,append,h} = 3", M["api_calls"](CALLS), 3.0)
check("api_calls: no calls = 0", M["api_calls"](FLAT), 0.0)

# ---------------------------------------------------------------- comment density
# '#' comment lines per 100 source lines. 2 comment lines, 2 source lines -> 100.
check("comment_density: no comments = 0", M["comment_density"](FLAT), 0.0)
check("comment_density: 2 comments / 2 sloc = 100",
      M["comment_density"]("# one\n# two\ndef f():\n    return 1\n"), 100.0)

# ---------------------------------------------------------------- AST similarity
# tree-edit distance on node TYPES (identifiers/literals ignored); 100 = identical.
A = "def f(a, b):\n    return a + b\n"
A_RENAMED = "def g(x, y):\n    return x + y\n"                 # same shape, different names
A_MINUS = "def f(a, b):\n    return a - b\n"                   # one operator changed
BRANCHED = "def f(a, b):\n    if a:\n        return a\n    return b\n"   # extra structure
check("ast_similarity: identical = 100", M["ast_similarity"](A, A), 100.0)
check("ast_similarity: rename only = 100 (identifiers ignored)", M["ast_similarity"](A, A_RENAMED), 100.0)
cond("ast_similarity: an operator change scores below 100",
     0 < M["ast_similarity"](A_MINUS, A) < 100, f"{M['ast_similarity'](A_MINUS, A):.2f}")
cond("ast_similarity: a small change scores higher than a big one",
     M["ast_similarity"](A_MINUS, A) > M["ast_similarity"](BRANCHED, A),
     f"minus={M['ast_similarity'](A_MINUS, A):.1f} vs branched={M['ast_similarity'](BRANCHED, A):.1f}")

# ---------------------------------------------------------------- robustness
# unparseable code must degrade to None (via the _safe wrapper), never raise.
BAD = "def f(:\n    return\n"
for name in ("cognitive", "api_calls", "comment_density"):
    check(f"{name}: unparseable -> None", M[name](BAD), None)
check("ast_similarity: unparseable -> None", M["ast_similarity"](BAD, A), None)


def main():
    npass = sum(1 for ok, *_ in CHECKS if ok)
    for ok, desc, got, exp in CHECKS:
        line = f"  [{'PASS' if ok else 'FAIL'}] {desc}"
        if not ok:
            line += f"   (got {got!r}, expected {exp!r})"
        print(line)
    print(f"\n{npass}/{len(CHECKS)} checks passed")
    sys.exit(0 if npass == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
