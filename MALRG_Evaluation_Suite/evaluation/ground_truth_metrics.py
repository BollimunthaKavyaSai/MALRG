"""
Ground-truth metrics: compare each system review with the human-written review of the same topic.

Computed (no LLM calls, local MiniLM embeddings only):
    rouge1_f1, rougeL_f1      unigram / longest-common-subsequence F1 on lower-cased word tokens
    document_cosine           cosine of the mean MiniLM sentence embeddings of the two reviews
    gt_sentence_recall        share of human sentences whose best system sentence has cosine >= TAU
    system_precision          share of system sentences whose best human sentence has cosine >= TAU
    semantic_f1               harmonic mean of the two shares above
    key_term_recall           share of the human review's key terms found in the system review

Key terms = the 40 highest-weight TF-IDF unigrams/bigrams of the human review (IDF over the 10 human reviews).
Citation markers, headings and the References section are removed before scoring.

Inputs : runs/<topic-slug>/<CONFIG>_review.md   and   <human-dir>/topic_XX_*.md  (topic order = config.TOPICS)
Output : results/tables/ground_truth_computed.csv  (mean over topics, one row per configuration)

    python -m evaluation.ground_truth_metrics --human "human_literature_reviews_written by human" --configs M0,M1,M2,M3,M4,M5

The thresholds below are choices of this script, not taken from the reported table; the values it
produces are what the shipped code measures.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from malrg.retrieval import slugify  # noqa: E402

TAU = 0.70           # cosine threshold for a sentence to count as matched
N_KEY_TERMS = 40
CITE_RE = re.compile(r"\[\d+(?:[,–\-\s]*\d+)*\]")


def clean(md: str) -> str:
    md = re.split(r"\n#+\s*references\b", md, flags=re.I)[0]
    md = re.sub(r"^#+.*$", "", md, flags=re.M)
    md = CITE_RE.sub("", md)
    md = re.sub(r"[*_`>|]", " ", md)
    return re.sub(r"[ \t]+", " ", md).strip()


def sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.split()) >= 5]


def words(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_n(ref, hyp, n=1):
    def grams(t):
        return [tuple(t[i:i + n]) for i in range(len(t) - n + 1)]
    from collections import Counter
    r, h = Counter(grams(ref)), Counter(grams(hyp))
    ov = sum((r & h).values())
    if not ov:
        return 0.0
    p, rc = ov / max(1, sum(h.values())), ov / max(1, sum(r.values()))
    return 2 * p * rc / (p + rc)


def rouge_l(ref, hyp):
    if not ref or not hyp:
        return 0.0
    prev = [0] * (len(hyp) + 1)
    for a in ref:
        cur = [0]
        for j, b in enumerate(hyp, 1):
            cur.append(prev[j - 1] + 1 if a == b else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, r = lcs / len(hyp), lcs / len(ref)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="folder with topic_01_*.md ... topic_10_*.md")
    ap.add_argument("--configs", default="M0,M1,M2,M3,M4,M5")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    hdir = Path(a.human)
    hfiles = sorted(hdir.glob("topic_*.md"))
    assert len(hfiles) == len(config.TOPICS), f"{len(hfiles)} human files for {len(config.TOPICS)} topics"
    humans = [clean(f.read_text(encoding="utf-8")) for f in hfiles]
    tf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True).fit(humans)
    vocab = np.array(tf.get_feature_names_out())

    rows = []
    for cfg in a.configs.split(","):
        acc = {k: [] for k in ("rouge1_f1", "rougeL_f1", "document_cosine", "gt_sentence_recall",
                               "system_precision", "semantic_f1", "key_term_recall")}
        for i, topic in enumerate(config.TOPICS):
            f = config.RUNS_DIR / slugify(topic) / f"{cfg}_review.md"
            if not f.exists():
                continue
            sys_t, hum_t = clean(f.read_text(encoding="utf-8")), humans[i]
            sw, hw = words(sys_t), words(hum_t)
            acc["rouge1_f1"].append(rouge_n(hw, sw, 1))
            acc["rougeL_f1"].append(rouge_l(hw, sw))
            ss, hs = sentences(sys_t), sentences(hum_t)
            es = emb.encode(ss, normalize_embeddings=True)
            eh = emb.encode(hs, normalize_embeddings=True)
            ds, dh = es.mean(0), eh.mean(0)
            acc["document_cosine"].append(float(ds @ dh / (np.linalg.norm(ds) * np.linalg.norm(dh))))
            sim = eh @ es.T
            rec = float((sim.max(1) >= TAU).mean())
            pre = float((sim.max(0) >= TAU).mean())
            acc["gt_sentence_recall"].append(rec)
            acc["system_precision"].append(pre)
            acc["semantic_f1"].append(0.0 if rec + pre == 0 else 2 * rec * pre / (rec + pre))
            w = tf.transform([hum_t]).toarray()[0]
            keys = vocab[np.argsort(w)[::-1][:N_KEY_TERMS]]
            low = " " + " ".join(sw) + " "
            acc["key_term_recall"].append(float(np.mean([(" " + k + " ") in low for k in keys])))
        n = len(acc["rouge1_f1"])
        rows.append({"config": cfg, "n_topics": n, **{k: round(float(np.mean(v)), 3) if v else "" for k, v in acc.items()}})
        print(rows[-1])

    out = ROOT / "results" / "tables" / "ground_truth_computed.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("written", out)


if __name__ == "__main__":
    main()
