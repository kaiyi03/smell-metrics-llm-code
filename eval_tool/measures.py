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

from dataclasses import dataclass
from typing import Callable

from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze


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


def _ensure_wordnet():
    """METEOR needs the WordNet corpus; fetch it once if it isn't already there."""
    for res, pkg in (("corpora/wordnet", "wordnet"), ("corpora/omw-1.4", "omw-1.4")):
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_wordnet()


def _bleu(code, ref):
    return sacrebleu.sentence_bleu(code, [ref]).score            # already 0..100

def _chrf(code, ref):
    return sacrebleu.sentence_chrf(code, [ref]).score            # already 0..100

def _rouge_l(code, ref):
    return _rouge.score(ref, code)["rougeL"].fmeasure * 100

def _codebleu(code, ref):
    return calc_codebleu([ref], [code], lang="python")["codebleu"] * 100

def _meteor(code, ref):
    return single_meteor_score(_word.findall(ref), _word.findall(code)) * 100


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
]


# The full panel. Fluency (perplexity) can be appended here later.
PANEL = list(STRUCTURAL) + list(SIMILARITY)
