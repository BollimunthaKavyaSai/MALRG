"""
M6 = M5 + Extractive Claim Filter (post-processing of the finished M5 review, no new LLM calls).

Every claim sentence of the M5 review is re-verified against the source passages with the
*critic* NLI model (cross-encoder/nli-deberta-v3-base, a different model from the DeBERTa-v3
model used for evaluation). A claim that is not entailed is
  1. replaced by the most similar sentence of its cited source papers (an extractive rewrite that
     keeps the citation), or, when no sufficiently similar source sentence exists,
  2. removed, unless it is the only sentence that cites a source (coverage guard).

    python make_m6.py            # build + evaluate (NLI metrics only) for all topics
    python make_m6.py --topics 1 # first topic only
"""

import argparse
import json
import re
import time

import numpy as np

import config
from evaluation.nli import PassageIndex, get_verifier, verify_claim
from evaluation.metrics import CLAIM_RE, META_RE
from malrg.retrieval import benchmark_path, slugify
from malrg.text_utils import (is_reference_heading, parse_citations, split_sentences,
                              strip_citations)

KEEP_ENTAIL = 0.70        # critic entailment needed to keep an original sentence
MIN_SIMILARITY = 0.45     # source sentence must be at least this similar to the claim to replace it


def _is_claim(sentence):
    claim = strip_citations(sentence)
    return bool(parse_citations(sentence)) or (bool(CLAIM_RE.search(claim)) and not META_RE.search(claim))


def _best_source_sentence(claim, refs, index):
    """Most similar individual sentence of the cited papers (or of all papers when uncited)."""
    from malrg.retrieval import embedder
    if not hasattr(index, "sentences"):
        index._build_sentences()
    if not index.sentences:
        return None
    v = embedder().encode([claim], normalize_embeddings=True)[0]
    sims = index.sent_emb @ v
    if refs:
        sims = np.where(np.isin(index.sent_ref, list(refs)), sims, -1)
    i = int(np.argmax(sims))
    if sims[i] < 0:
        return None
    return int(index.sent_ref[i]), index.sentences[i][1], float(sims[i])


def _clean_source_sentence(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\[\d+(?:[,–\-\s]*\d+)*\]", "", text)          # drop the source's own citations
    text = re.sub(r"\([A-Z][A-Za-z\-]+(?: et al\.)?,? \d{4}[a-z]?\)", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ;,")
    if text and text[-1] not in ".!?":
        text += "."
    return text


def filter_review(review_md, index, verifier, n_papers):
    stats = {"claims": 0, "kept": 0, "replaced": 0, "deleted": 0, "kept_for_coverage": 0}
    # citation counts, used by the coverage guard
    counts = {r: 0 for r in range(1, n_papers + 1)}
    for line in review_md.splitlines():
        for s in split_sentences(line):
            for r in set(parse_citations(s)):
                if r in counts:
                    counts[r] += 1

    out_lines, in_refs, seen = [], False, set()
    for line in review_md.splitlines():
        m = re.match(r"^\s{0,3}(#{1,4})\s+(.*)$", line)
        if m:
            in_refs = is_reference_heading(m.group(2))
            out_lines.append(line)
            continue
        stripped = line.strip()
        if in_refs or not stripped or stripped.startswith("|") or re.match(r"^[-*_]{3,}$", stripped):
            out_lines.append(line)
            continue
        prefix = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+)", line)
        pre = prefix.group(1) if prefix else ""
        body = line[len(pre):]
        new_sents = []
        for s in split_sentences(body):
            plain = re.sub(r"[*`]+", "", s)
            if len(plain.split()) < 5 or not _is_claim(plain):
                new_sents.append(s)
                continue
            stats["claims"] += 1
            refs = [r for r in parse_citations(plain) if 1 <= r <= n_papers]
            claim = strip_citations(plain)
            keep = False
            if refs:
                keep = verify_claim(claim, refs, index, verifier)["entail"] >= KEEP_ENTAIL
            if keep:
                stats["kept"] += 1
                new_sents.append(s)
                continue
            cand = _best_source_sentence(claim, refs or None, index)
            sole = any(counts.get(r, 0) <= 1 for r in refs)
            if cand and (cand[2] >= MIN_SIMILARITY or sole):
                ref, text, _ = cand
                text = _clean_source_sentence(text)
                new = f"{text[:-1]} [{ref}]{text[-1]}" if text else ""
                key = re.sub(r"\W+", " ", new.lower())
                if new and key not in seen:
                    seen.add(key)
                    new_sents.append(new)
                    stats["replaced"] += 1
                    for r in refs:
                        counts[r] = counts.get(r, 1) - 1
                    counts[ref] = counts.get(ref, 0) + 1
                    continue
            stats["deleted"] += 1
            for r in refs:
                counts[r] = counts.get(r, 1) - 1
        if new_sents:
            out_lines.append(pre + " ".join(new_sents))
    md = "\n".join(out_lines)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md, stats


def build(topic, log=print, base="M5", out="M6"):
    d = config.RUNS_DIR / slugify(topic)
    m5 = json.loads((d / f"{base}.json").read_text(encoding="utf-8"))
    bench = json.loads(benchmark_path(topic).read_text(encoding="utf-8"))
    index = PassageIndex(bench["papers"])
    t0 = time.time()
    md, stats = filter_review(m5["review_md"], index, get_verifier("critic"), len(bench["papers"]))
    rec = {**m5, "config_id": out, "config_name": f"{out}: {base} + Extractive Claim Filter",
           "agent_chain": m5["agent_chain"] + ["extractive_claim_filter"], "review_md": md,
           "filter_stats": stats, "filter_time_s": round(time.time() - t0, 1),
           "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / f"{out}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / f"{out}_review.md").write_text(md, encoding="utf-8")
    for f in (f"{out}_eval.json",):
        p = d / f
        if p.exists():
            p.unlink()                      # force a fresh evaluation
    log(f"  {out} {topic[:50]}: {stats} words {len(m5['review_md'].split())} -> {len(md.split())}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=int, default=len(config.TOPICS))
    ap.add_argument("--base", default="M5", help="config to filter, e.g. M1 for a strong RAG baseline")
    ap.add_argument("--out", default="M6", help="id of the filtered config, e.g. B3")
    args = ap.parse_args()
    from evaluation import runner
    rows = []
    for topic in config.TOPICS[:args.topics]:
        build(topic, log=print, base=args.base, out=args.out)
        res = runner.evaluate_run(topic, args.out, log=print, use_judge=False)
        m5 = json.loads((config.RUNS_DIR / slugify(topic) / f"{args.base}_eval.json").read_text(encoding="utf-8"))["nli"]
        rows.append((topic, m5, res["nli"]))
    keys = ["citation_groundedness", "hallucination_rate", "fabricated_reference_rate", "source_coverage",
            "multi_source_synthesis", "uncited_claim_rate", "contradiction_rate"]
    print("\nmetric                      M5      M6")
    for k in keys:
        a = np.nanmean([r[1][k] for r in rows])
        b = np.nanmean([r[2][k] for r in rows])
        print(f"{k:28s}{a:6.3f}  {b:6.3f}")


if __name__ == "__main__":
    main()
