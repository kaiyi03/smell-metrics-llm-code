"""
The measure panel for the evaluation tool.

A "measure" is one number we can compute about a piece of code. They come in
families:

  structural  (reference-free)  -- just needs the code. Radon complexity,
                                   maintainability, size. Higher complexity /
                                   lower maintainability = worse.
  similarity  (reference-based) -- needs a clean reference to compare against:
                                   BLEU, CodeBLEU, ROUGE ...
  fluency     (reference-free)  -- a language model's perplexity (not yet built)
  correctness (execution)       -- run the task's tests

We deliberately do NOT put Pylint / Ruff / jscpd in here. Those tools built the
dataset labels, so grading them against those same labels would be circular.
They are the "operational definition" of each smell, not a measure under test.
This panel is the set of INDEPENDENT measures we are actually evaluating.

Every measure returns a float, or None if it can't be computed (e.g. code that
won't parse). None-valued results are skipped in the stats -- never counted as 0.
"""

import ast
import difflib
import textwrap
from dataclasses import dataclass
from typing import Callable

from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze

try:                                       # cognitive complexity (SonarSource algorithm)
    from cognitive_complexity.api import get_cognitive_complexity
except Exception:                          # keep the panel importable without the lib
    get_cognitive_complexity = None


@dataclass
class Measure:
    name: str
    family: str        # 'structural' | 'similarity' | 'fluency' | 'correctness'
    worse: str         # 'up' or 'down' -- the direction that signals a smell
    fn: Callable       # reference-free: fn(code) ; reference-based: fn(code, ref)
    needs_ref: bool
    blurb: str


def _safe(fn):
    """Wrap a raw measure so any failure (unparseable code, etc.) returns None."""
    def wrapped(*args):
        try:
            v = fn(*args)
            return None if v is None else float(v)
        except Exception:
            return None
    return wrapped


# ---- structural measures (reference-free) -----------------------------------

def _sloc(code):
    return analyze(code).sloc                    # source lines of code

def _cyclomatic(code):
    blocks = cc_visit(code)                       # one block per function/class
    return sum(b.complexity for b in blocks) if blocks else 1.0

def _maintainability(code):
    return mi_visit(code, True)                   # 0..100, higher = better

def _halstead_volume(code):
    return h_visit(code).total.volume             # program size in operators/operands

def _halstead_difficulty(code):
    return h_visit(code).total.difficulty         # how hard to read/write

def _halstead_effort(code):
    return h_visit(code).total.effort             # volume x difficulty

def _cognitive(code):
    """SonarSource cognitive complexity -- like cyclomatic, but it adds a nesting
    penalty so deeply nested branches cost more than flat ones. Summed over every
    function in the snippet; a module with no function is wrapped so its top-level
    control flow still counts."""
    if get_cognitive_complexity is None:
        return None
    tree = ast.parse(code)
    fns = [n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if fns:
        return float(sum(get_cognitive_complexity(fn) for fn in fns))
    wrapped = ast.parse("def _m():\n" + textwrap.indent(code, "    "))
    return float(get_cognitive_complexity(wrapped.body[0]))

def _comment_density(code):
    """Comment-to-code ratio: '#' comment lines per 100 source lines (radon raw
    metrics). A documentation signal -- it barely moves for the injected smells,
    which don't touch comments, but profiles how well real / generated code is
    commented."""
    m = analyze(code)
    return 100.0 * m.comments / m.sloc if m.sloc else 0.0

def _api_calls(code):
    """Function / API usage: how many DISTINCT functions or methods the code calls
    (AST call expressions, by callee name). A usage profile -- does the code lean on
    library calls or inline everything -- rather than a smell detector."""
    tree = ast.parse(code)
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return float(len(names))


STRUCTURAL = [
    Measure("sloc",                "structural", "up",   _safe(_sloc),
            False, "source lines of code -- raw size"),
    Measure("cyclomatic",          "structural", "up",   _safe(_cyclomatic),
            False, "cyclomatic complexity -- number of decision paths"),
    Measure("maintainability",     "structural", "down", _safe(_maintainability),
            False, "Radon maintainability index 0-100 (higher is better)"),
    Measure("halstead_volume",     "structural", "up",   _safe(_halstead_volume),
            False, "Halstead volume -- program size"),
    Measure("halstead_difficulty", "structural", "up",   _safe(_halstead_difficulty),
            False, "Halstead difficulty -- reading/writing effort"),
    Measure("halstead_effort",     "structural", "up",   _safe(_halstead_effort),
            False, "Halstead effort -- volume x difficulty"),
    Measure("cognitive",           "structural", "up",   _safe(_cognitive),
            False, "cognitive complexity -- cyclomatic plus a nesting penalty"),
    Measure("comment_density",     "structural", "down", _safe(_comment_density),
            False, "comment lines per 100 source lines (documentation profile)"),
    Measure("api_calls",           "structural", "down", _safe(_api_calls),
            False, "distinct functions / APIs called (usage profile, not a smell)"),
]


# ---- similarity measures (reference-based) ----------------------------------
# Each scores a candidate against a clean reference. All are rescaled to 0..100
# where 100 = identical to the reference; lower = the candidate diverges more.
# On the injected set the reference is the clean twin, so clean-vs-clean is 100
# by construction and we only score the smelly side. Later, the reference is the
# task's canonical solution and the candidate is the model's output.

import re                                            # noqa: E402
import sacrebleu                                     # noqa: E402
import nltk                                          # noqa: E402
from codebleu import calc_codebleu                   # noqa: E402
from rouge_score import rouge_scorer                 # noqa: E402
from nltk.translate.meteor_score import single_meteor_score  # noqa: E402

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
_word = re.compile(r"\w+|\S")


def _wordnet_available():
    """METEOR needs the WordNet corpus. Check for it WITHOUT triggering a network
    download at import time -- a failed download hangs the process (offline ARC, a
    sandbox, no internet). If WordNet is genuinely absent, METEOR degrades to None and
    the other five similarity measures carry on. Install it once, ahead of time, with:
        python -m nltk.downloader wordnet omw-1.4"""
    for res in ("corpora/wordnet", "corpora/omw-1.4"):
        try:
            nltk.data.find(res)
        except LookupError:
            return False
    return True


_WORDNET = _wordnet_available()


def _bleu(code, ref):
    return sacrebleu.sentence_bleu(code, [ref]).score            # already 0..100

def _chrf(code, ref):
    return sacrebleu.sentence_chrf(code, [ref]).score            # already 0..100

def _rouge_l(code, ref):
    return _rouge.score(ref, code)["rougeL"].fmeasure * 100

def _codebleu(code, ref):
    return calc_codebleu([ref], [code], lang="python")["codebleu"] * 100

def _meteor(code, ref):
    if not _WORDNET:                                   # corpus missing -> degrade, don't crash
        return None
    return single_meteor_score(_word.findall(ref), _word.findall(code)) * 100

def _ast_seq(code):
    """Pre-order (depth-first) sequence of AST node types -- the code's structural
    skeleton, with identifiers and literal values dropped."""
    seq = []

    def visit(node):
        seq.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(code))
    return seq

def _ast_similarity(code, ref):
    """Structure-aware similarity: how much the candidate's AST skeleton matches the
    reference's. Both are flattened to their pre-order node-type sequences (identifiers
    and literals ignored), then compared with difflib's longest-matching-blocks ratio.
    100 = identical structure; lower = more structural change. Complements the
    token-based measures, which a rename or reformat can fool but a real structural
    change cannot."""
    a, b = _ast_seq(code), _ast_seq(ref)
    if not a and not b:
        return 100.0
    return 100.0 * difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


SIMILARITY = [
    Measure("bleu",     "similarity", "down", _safe(_bleu),     True,
            "BLEU vs clean reference (word n-grams)"),
    Measure("chrf",     "similarity", "down", _safe(_chrf),     True,
            "chrF vs clean reference (character n-grams)"),
    Measure("rouge_l",  "similarity", "down", _safe(_rouge_l),  True,
            "ROUGE-L longest-common-subsequence overlap"),
    Measure("meteor",   "similarity", "down", _safe(_meteor),   True,
            "METEOR: unigram match with stems/synonyms + word-order penalty"),
    Measure("codebleu", "similarity", "down", _safe(_codebleu), True,
            "CodeBLEU: code-aware, includes AST + dataflow match"),
    Measure("ast_similarity", "similarity", "down", _safe(_ast_similarity), True,
            "AST skeleton similarity (structure-aware, identifiers ignored)"),
]


# The full panel. Fluency (perplexity) can be appended here later.
PANEL = list(STRUCTURAL) + list(SIMILARITY)
