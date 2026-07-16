"""Model-based measures -- the GPU phase (perplexity, CodeBERTScore, BERTScore).

These need PyTorch and a GPU, so they run on a GPU node (Oxford ARC), not on the
laptop or in the dashboard. measures.py imports this inside a try/except and only
adds these measures when `AVAILABLE` is true (a CUDA GPU is present), so on every
other machine the panel stays exactly as it was -- CPU-only, torch-free.

The three measures, in the panel's own vocabulary:
  * perplexity     (fluency, reference-free)  -- a code language model's perplexity on
                    the snippet. Higher = the model finds the code more surprising /
                    less natural, which we expect for smelly code. worse = up.
  * codebert_score (similarity, reference-based) -- CodeBERTScore F1: cosine similarity
                    of CodeBERT token embeddings, aligned candidate<->reference. A
                    code-aware semantic complement to token-overlap BLEU/CodeBLEU.
  * bertscore      (similarity, reference-based) -- standard BERTScore F1 (RoBERTa
                    embeddings). The general-language semantic-similarity baseline.

Models load LAZILY on first use and are cached module-level, so a full run over the
benchmark loads each model once, not once per snippet.

Env knobs:
  PPL_MODEL          causal LM for perplexity  (default: Salesforce/codegen-350M-mono,
                     a small code model; set to a bigger one, e.g. the Qwen coder, for
                     a stronger signal)
  BERT_MODEL         model for BERTScore        (default: roberta-large via lang="en")
  GPU_MEASURES_CPU   set to force these on even without CUDA (slow; for a smoke test)
"""

import os

import torch   # if torch is absent this raises -> measures.py keeps the CPU-only panel

AVAILABLE = torch.cuda.is_available() or bool(os.environ.get("GPU_MEASURES_CPU"))
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_lm = {}        # perplexity model cache
_scorer = {}    # BERTScorer cache


# ---- perplexity (reference-free) --------------------------------------------

def _load_lm():
    if "model" not in _lm:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = os.environ.get("PPL_MODEL", "Salesforce/codegen-350M-mono")
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).to(_DEVICE).eval()
        _lm["model"], _lm["tok"] = model, tok
    return _lm["model"], _lm["tok"]


def _perplexity(code):
    model, tok = _load_lm()
    ids = tok(code, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(_DEVICE)
    if ids.size(1) < 2:                     # need at least one predicted token
        return None
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return float(torch.exp(loss).item())    # perplexity = exp(mean token cross-entropy)


# ---- BERTScore (reference-based) --------------------------------------------

def _bertscorer():
    if "s" not in _scorer:
        from bert_score import BERTScorer
        name = os.environ.get("BERT_MODEL")
        if name:
            _scorer["s"] = BERTScorer(model_type=name, device=_DEVICE)
        else:
            _scorer["s"] = BERTScorer(lang="en", device=_DEVICE)   # -> roberta-large
    return _scorer["s"]


def _bertscore(code, ref):
    _p, _r, f1 = _bertscorer().score([code], [ref])
    return float(f1[0].item()) * 100


# ---- CodeBERTScore (reference-based) ----------------------------------------

def _codebert_score(code, ref):
    import code_bert_score                  # caches its model internally across calls
    _p, _r, f1, _f3 = code_bert_score.score(cands=[code], refs=[ref], lang="python")
    return float(f1[0].item()) * 100


def measures(Measure, safe):
    """Built lazily and passed measures.Measure + measures._safe, so this module never
    imports measures.py (avoids a circular import)."""
    return [
        Measure("perplexity", "fluency", "up", safe(_perplexity), False,
                "code LM perplexity -- how unnatural/surprising the code is (higher = worse)"),
        Measure("codebert_score", "similarity", "down", safe(_codebert_score), True,
                "CodeBERTScore F1 -- code-aware embedding similarity to the reference"),
        Measure("bertscore", "similarity", "down", safe(_bertscore), True,
                "BERTScore F1 -- contextual-embedding similarity to the reference"),
    ]
