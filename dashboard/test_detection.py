"""Does the dashboard detect smells in ARBITRARY code -- not just dataset snippets?

Every example here is hand-written (nothing from the benchmark) and each is checked
against the dashboard's own detector, detect_labeled(). It covers all 12 smells, a
few in non-obvious shapes (a class method, module-level code), clean code that must
stay clean (no false positives), multi-smell code, and broken input.

Run:  .venv/Scripts/python dashboard/test_detection.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app import detect_labeled   # noqa: E402  (the exact function the dashboard calls)

CHECKS = []


def want(name, code, smell):
    """detect_labeled(code) must include `smell`."""
    found, problems = detect_labeled(code)
    CHECKS.append((smell in found, f"{name}: detects {smell}",
                   f"found={found} problems={problems}"))


def want_clean(name, code):
    found, problems = detect_labeled(code)
    CHECKS.append((found == [] and not problems, f"{name}: stays clean",
                   f"found={found} problems={problems}"))


# ---- one hand-written example per smell (thresholds crossed on purpose) ----
want("plain mutable default", "def f(x, acc=[]):\n    acc.append(x)\n    return acc\n", "mutable_default")
want("broad except", "def f():\n    try:\n        risky()\n    except Exception:\n        pass\n", "broad_except")
want("unused import", "import os\n\ndef f(a):\n    return a + 1\n", "dead_code")
want("unused variable", "def f(a):\n    leftover = 99\n    return a + 1\n", "dead_code")
want("magic number in compare", "def f(x):\n    return x == 42\n", "magic_number")
want("six parameters", "def f(a, b, c, d, e, g):\n    return a\n", "long_parameter_list")
want("six boolean terms",
     "def f(o):\n    if o.a and o.b and o.c and o.d and o.e and o.f:\n        return 1\n    return 0\n",
     "complex_conditional")
want("six-deep nesting",
     "def f(a):\n" + "".join("    " * (i + 1) + f"if a > {i}:\n" for i in range(6))
     + "    " * 7 + "return a\n",
     "deep_nesting")
want("fifty-plus statements",
     "def f():\n" + "".join(f"    a{i} = {i}\n" for i in range(60)) + "    return a0\n",
     "long_method")
want("manual append-with-transform loop",
     "def f(xs):\n    out = []\n    for x in xs:\n        out.append(x * 2)\n    return out\n",
     "inefficient_loop")
want("manual list copy loop",
     "def f(xs):\n    out = []\n    for x in xs:\n        out.append(x)\n    return out\n",
     "inefficient_copy")
want("try inside a loop",
     "def f(xs):\n    for x in xs:\n        try:\n            g(x)\n        except Exception:\n            pass\n",
     "perf_try_in_loop")
want("within-snippet duplicate block (>=5 lines, >=25 tokens)",
     "def process_first(data):\n    result = []\n    for item in data:\n        value = item * 2\n"
     "        value = value + 10\n        value = value - 3\n        if value > 0:\n"
     "            result.append(value)\n    return result\n\n"
     "def process_second(data):\n    result = []\n    for item in data:\n        value = item * 2\n"
     "        value = value + 10\n        value = value - 3\n        if value > 0:\n"
     "            result.append(value)\n    return result\n",
     "duplicate_code")

# ---- non-obvious code shapes: it must still work off the beaten path ----
want("smell inside a class method",
     "class C:\n    def m(self, x, acc=[]):\n        acc.append(x)\n        return acc\n", "mutable_default")
want("module-level broad except",
     "try:\n    x = risky()\nexcept Exception:\n    x = None\n", "broad_except")
want("async function broad except",
     "async def f():\n    try:\n        await g()\n    except Exception:\n        return None\n", "broad_except")

# ---- clean code must NOT be flagged (no false positives) ----
want_clean("simple add", "def add(a, b):\n    return a + b\n")
want_clean("three params, shallow, comprehension",
           "def f(a, b, c):\n    if a:\n        return [x for x in b if x > c]\n    return []\n")

# ---- multi-smell code: several at once ----
_multi = ("def f(x, acc=[]):\n    unused = 1\n    try:\n        if x == 7:\n            acc.append(x)\n"
          "    except Exception:\n        pass\n    return acc\n")
_found, _ = detect_labeled(_multi)
CHECKS.append(({"mutable_default", "dead_code", "broad_except", "magic_number"} <= set(_found),
               "multi-smell: mutable_default + dead_code + broad_except + magic_number all found",
               f"found={_found}"))

# ---- broken input must be reported, not silently 'clean' ----
_found, _problems = detect_labeled("def f(:\n    return\n")
CHECKS.append((_found == [] and bool(_problems), "syntax error: reported as a problem, not 'clean'",
               f"found={_found} problems={_problems}"))


def main():
    npass = sum(1 for ok, *_ in CHECKS if ok)
    for ok, desc, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}" + ("" if ok else f"   -> {detail}"))
    print(f"\n{npass}/{len(CHECKS)} checks passed")
    sys.exit(0 if npass == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
