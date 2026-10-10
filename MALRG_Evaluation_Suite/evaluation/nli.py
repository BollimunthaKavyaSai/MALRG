"""
Evidence-anchored claim verification with Natural Language Inference.

A claim (review sentence, citations stripped) is checked against passages of the
paper(s) it cites. Passages are pre-selected by MiniLM similarity (top-k per paper)
and then scored by a DeBERTa-v3 NLI cross-encoder. A claim is SUPPORTED when
P(entailment) >= threshold for some cited passage (or for the concatenation of the
best passage of each cited paper, which handles multi-source claims).

Two independent verifiers are exposed:
  * "eval"   - MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (used for all reported metrics)
  * "critic" - cross-encoder/nli-deberta-v3-base       (used inside the feedback loop)
Using a different model for the critic than for evaluation avoids optimising the
system against the very instrument that measures it.
"""

import os
from functools import lru_cache

import numpy as np
import torch

import config
from malrg.retrieval import embedder

CRITIC_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


class NLIVerifier:
    def __init__(self, model_name):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).float().eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.ent = labels["entailment"]
        self.con = labels["contradiction"]
        torch.set_num_threads(min(4, os.cpu_count() or 4))  # fastest on this CPU (profiled)
        self._cache = {}

    @torch.inference_mode()
    def score(self, pairs, batch_size=16):
        """pairs: [(premise, hypothesis)] -> np.array [[p_entail, p_contra], ...]"""
        if len(self._cache) > 20000:      # bound memory over long multi-hour attempts
            self._cache.clear()
        out = [None] * len(pairs)
        todo = []
        for i, p in enumerate(pairs):
            key = (p[0], p[1])
            if key in self._cache:
                out[i] = self._cache[key]
            else:
                todo.append(i)
        # length-sorted batching minimizes padding (large CPU speed-up)
        todo.sort(key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        for s in range(0, len(todo), batch_size):
            idx = todo[s:s + batch_size]
            prem = [pairs[i][0] for i in idx]
            hyp = [pairs[i][1] for i in idx]
            enc = self.tok(prem, hyp, truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt")
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).numpy()
            for j, i in enumerate(idx):
                val = (float(probs[j, self.ent]), float(probs[j, self.con]))
                self._cache[(pairs[i][0], pairs[i][1])] = val
                out[i] = val
        return np.array(out) if out else np.zeros((0, 2))


@lru_cache(maxsize=2)
def get_verifier(kind="eval"):
    return NLIVerifier(config.NLI_MODEL if kind == "eval" else CRITIC_NLI_MODEL)


class PassageIndex:
    """Embedding index over the full-text passages of a topic's source papers."""

    def __init__(self, papers):
        self.papers = {p["ref_id"]: p for p in papers}
        self.passages = []          # (ref_id, text)
        for p in papers:
            for t in p["passages"]:
                self.passages.append((p["ref_id"], t))
            # the abstract is also valid evidence
            self.passages.append((p["ref_id"], p.get("abstract", "")))
        texts = [t for _, t in self.passages]
        self.emb = embedder().encode(texts, normalize_embeddings=True, batch_size=64) if texts else None
        self.ref_of = np.array([r for r, _ in self.passages])

    def _build_sentences(self):
        from malrg.text_utils import split_sentences
        self.sentences = []
        seen = set()
        for ref, text in self.passages:
            for s in split_sentences(text):
                key = (ref, s[:80])
                if 8 <= len(s.split()) <= 60 and key not in seen:
                    seen.add(key)
                    self.sentences.append((ref, s))
        self.sent_emb = embedder().encode([s for _, s in self.sentences], normalize_embeddings=True,
                                          batch_size=64)
        self.sent_ref = np.array([r for r, _ in self.sentences])

    def top_sentences(self, claim, ref_ids=None, k=3):
        """Most similar individual evidence sentences (used to anchor the feedback reviser)."""
        if not hasattr(self, "sentences"):
            self._build_sentences()
        if not self.sentences:
            return []
        v = embedder().encode([claim], normalize_embeddings=True)[0]
        sims = self.sent_emb @ v
        if ref_ids:
            sims = np.where(np.isin(self.sent_ref, list(ref_ids)), sims, -1)
        order = np.argsort(-sims)[:k]
        return [(int(self.sent_ref[i]), self.sentences[i][1]) for i in order if sims[i] > -1]

    def top_passages(self, claim, ref_ids=None, k=config.NLI_TOP_K_PASSAGES, claim_vec=None):
        if self.emb is None:
            return []
        if claim_vec is None:
            claim_vec = embedder().encode([claim], normalize_embeddings=True)[0]
        sims = self.emb @ claim_vec
        if ref_ids is not None:
            mask = np.isin(self.ref_of, list(ref_ids))
            sims = np.where(mask, sims, -1)
        order = np.argsort(-sims)[:k * (len(ref_ids) if ref_ids else 1)]
        return [(int(self.ref_of[i]), self.passages[i][1], float(sims[i])) for i in order if sims[i] > -1]


def verify_claim(claim, ref_ids, index, verifier, k=config.NLI_TOP_K_PASSAGES):
    """Verify `claim` against the passages of `ref_ids`. Returns a detail dict."""
    valid = [r for r in ref_ids if r in index.papers]
    if not valid:
        return {"supported": False, "entail": 0.0, "contra": 0.0, "best_ref": None,
                "reason": "no_valid_source"}
    claim_vec = embedder().encode([claim], normalize_embeddings=True)[0]
    pairs, meta = [], []
    best_by_ref = {}
    for r in valid:
        cands = index.top_passages(claim, [r], k=k, claim_vec=claim_vec)[:k]
        # premises: the single best passage + an evidence window of the top passages
        # (the window subsumes the other single passages, halving NLI cost on CPU)
        for ref, text, _ in cands[:1]:
            pairs.append((text, claim))
            meta.append(ref)
        if len(cands) > 1:
            # evidence window: synthesis sentences often combine facts that a paper states
            # in different passages, so the top passages are also scored as one premise
            pairs.append((" ".join(t for _, t, _ in cands[:config.NLI_WINDOW_PASSAGES]), claim))
            meta.append(r)
        if cands:
            best_by_ref[r] = cands[0][1]
    if len(valid) > 1 and len(best_by_ref) > 1:
        pairs.append((" ".join(best_by_ref.values()), claim))
        meta.append("joint")
    scores = verifier.score(pairs)
    if len(scores) == 0:
        return {"supported": False, "entail": 0.0, "contra": 0.0, "best_ref": None,
                "reason": "no_passages"}
    i = int(np.argmax(scores[:, 0]))
    ent = float(scores[i, 0])
    con = float(scores[:, 1].max())
    per_ref = {}
    for (e, _), m in zip(scores, meta):
        if m != "joint":
            per_ref[m] = max(per_ref.get(m, 0.0), float(e))
    return {
        "supported": ent >= config.NLI_ENTAIL_THRESHOLD,
        "entail": round(ent, 4),
        "contra": round(con, 4),
        "contradicted": ent < config.NLI_ENTAIL_THRESHOLD and con >= config.NLI_CONTRA_THRESHOLD,
        "best_ref": meta[i],
        "per_ref_entail": per_ref,
        "evidence": pairs[i][0][:600],
    }


def best_support_any(claim, index, verifier, k=6):
    """Search all papers for a passage that entails `claim` (used for re-citation)."""
    cands = index.top_passages(claim, None, k=k)
    if not cands:
        return None
    pairs = [(cands[0][1], claim)]
    refs = [cands[0][0]]
    by_ref = {}
    for r, t, _ in cands:
        by_ref.setdefault(r, []).append(t)
    for r, ts in list(by_ref.items())[:2]:     # windows for the two most similar papers
        if len(ts) > 1:
            pairs.append((" ".join(ts[:config.NLI_WINDOW_PASSAGES]), claim))
            refs.append(r)
    scores = verifier.score(pairs)
    i = int(np.argmax(scores[:, 0]))
    return {"ref": refs[i], "entail": float(scores[i, 0]), "passage": pairs[i][0]}
