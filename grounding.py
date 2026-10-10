"""
Evidence-grounding toolkit shared by the feedback loop (review_feedback.py)
and the automatic evaluator (evaluation/metrics.py).

* Review parsing: sections, sentences, numeric citations [1], [2, 4], [1]-[3].
* EvidenceIndex: passage store built from each source paper's abstract,
  extracted full text and structured analysis, with MiniLM retrieval.
* NLIScorer: cross-encoder natural-language-inference model that returns
  entailment / contradiction probabilities for (evidence, claim) pairs.

The feedback loop and the evaluator deliberately use DIFFERENT NLI models
(MiniLM2 vs. DeBERTa-v3) so that the system is not optimised against the
exact judge that later scores it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FEEDBACK_NLI_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"
JUDGE_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
PASSAGE_CHARS = 700


# Speed-up: once all three models (MiniLM embeddings and the two NLI models) are in the local
# Hugging Face cache, set the offline flags so no slow internet checks happen at every start.
def _use_offline_hub_if_cached() -> None:
    """Skip slow Hugging Face network checks once every model is cached."""
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    needed = [EMBEDDING_MODEL, FEEDBACK_NLI_MODEL, JUDGE_NLI_MODEL]
    if all((hub / ("models--" + name.replace("/", "--"))).exists() for name in needed):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_use_offline_hub_if_cached()

CERTAINTY_PHRASES = (
    "empirically proven",
    "definitively demonstrates",
    "definitively proves",
    "conclusively",
    "undeniably",
    "it is proven",
    "proves that",
    "guarantees",
    "always outperforms",
    "universally",
    "without exception",
)

COMPARATIVE_MARKERS = (
    "whereas",
    "while ",
    "in contrast",
    "by contrast",
    "compared",
    "comparison",
    "unlike",
    "similarly",
    "likewise",
    "both ",
    "outperform",
    "differ",
    "consistent with",
    "contrary",
    "on the other hand",
    "in common",
    "complement",
    "trade-off",
    "tradeoff",
)

# ----------------------------------------------------------------------
# Review parsing
# ----------------------------------------------------------------------

_ABBREVIATIONS = (
    "et al.", "e.g.", "i.e.", "vs.", "Fig.", "fig.", "Eq.", "eq.", "approx.",
    "etc.", "cf.", "No.", "no.", "Vol.", "pp.", "Dr.", "Prof.", "resp.",
)

CITATION_PATTERN = re.compile(r"\[(\s*\d+\s*(?:[,;–\-]\s*\d+\s*)*)\]")


# Splits a review into (body text, references text) at the line '## References'.
# If there is no References heading, the whole text is the body.
def split_body_and_references(review: str) -> Tuple[str, str]:
    match = re.search(r"(?im)^#{1,6}\s*references\s*$", review or "")
    if not match:
        return review or "", ""
    return review[: match.start()], review[match.end():]


# Cuts the review body into (heading, text) pieces by scanning line by line and starting a new piece
# at every markdown heading. Text before the first heading has an empty heading.
def split_sections(body: str) -> List[Tuple[str, str]]:
    """Return [(heading, text)] for the review body (heading '' = preamble)."""
    sections: List[Tuple[str, str]] = []
    current_heading = ""
    current_lines: List[str] = []
    for line in (body or "").splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if heading:
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = heading.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    return [(h, t) for h, t in sections if t]


# Breaks text into sentences without cutting at 'et al.', 'e.g.', 'Fig.' or decimal numbers.
# It hides those dots first, splits blocks at blank lines/bullets and sentence endings, attaches a lone
# citation like '[3].' to the previous sentence, then restores the hidden dots.
def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    protected = text
    for i, abbr in enumerate(_ABBREVIATIONS):
        protected = protected.replace(abbr, abbr.replace(".", f"§{i}§"))
    protected = re.sub(r"(\d)\.(\d)", r"\1¤\2", protected)

    sentences: List[str] = []
    for block in re.split(r"\n\s*\n|\n(?=\s*[-*•]\s)", protected):
        block = re.sub(r"^\s*[-*•]\s+", "", block.strip())
        block = re.sub(r"\s*\n\s*", " ", block)
        if not block:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[\[(\"“A-Z0-9])", block)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Attach a dangling citation such as "[3]." to the previous sentence.
            if sentences and re.fullmatch(r"(\[[\d,;\s–\-]+\]\s*)+[.;]?", part):
                sentences[-1] = sentences[-1].rstrip() + " " + part
                continue
            sentences.append(part)

    restored = []
    for sentence in sentences:
        for i, abbr in enumerate(_ABBREVIATIONS):
            sentence = sentence.replace(abbr.replace(".", f"§{i}§"), abbr)
        restored.append(sentence.replace("¤", "."))
    return restored


# Finds the paper numbers cited in a sentence: [1], [2, 4], [1-3] and [1]-[3] all work.
# Returns the unique numbers in order of appearance.
def extract_citations(sentence: str) -> List[int]:
    cited: List[int] = []
    for group in CITATION_PATTERN.findall(sentence or ""):
        for token in re.split(r"[,;]", group):
            token = token.strip()
            span = re.split(r"\s*[–\-]\s*", token)
            if len(span) == 2 and span[0].isdigit() and span[1].isdigit():
                lo, hi = int(span[0]), int(span[1])
                if 0 < hi - lo <= 20:
                    cited.extend(range(lo, hi + 1))
                    continue
            if token.isdigit():
                cited.append(int(token))
    # "[1]-[3]" style ranges
    for lo, hi in re.findall(r"\[(\d+)\]\s*[–\-]\s*\[(\d+)\]", sentence or ""):
        lo, hi = int(lo), int(hi)
        if 0 < hi - lo <= 20:
            cited.extend(range(lo, hi + 1))
    return list(dict.fromkeys(cited))


_NOUN_SLOT_WORDS = {
    "and", "both", "while", "whereas", "by", "of", "in", "from", "unlike", "than", "with",
    "as", "to", "contrast,", "however,", "similarly,", "meanwhile,", "study", "work",
}


# Inside strip_citations: when a citation follows an author name ('Maity & Saikia [1] adopt'),
# removes it because the name already is the subject of the sentence.
def _drop_citations_after_author_names(text: str) -> str:
    """'Maity & Saikia [1] adopt' -> 'Maity & Saikia adopt' (the name is already the subject)."""
    # Helper used by the regex: decides for each matched citation whether to delete it (after an author name)
    # or keep it (when it acts like a noun, e.g. 'Both [1] and [2]').
    def replace(match: "re.Match[str]") -> str:
        before = text[: match.start()].rstrip()
        previous = before.split()[-1] if before.split() else ""
        if previous.lower() in _NOUN_SLOT_WORDS:
            return match.group(0)
        if previous.endswith("al.") or (previous[:1].isupper() and previous.lower() not in {"both", "in"}):
            return ""
        return match.group(0)

    return CITATION_PATTERN.sub(replace, text)


# Removes citation marks from a sentence so it can be checked as a plain claim.
# Trailing citations are deleted; a citation that acts as a subject becomes 'the cited study';
# markdown symbols and extra spaces are cleaned. The grammar of the sentence stays correct.
def strip_citations(sentence: str) -> str:
    """Remove citation markers, keeping the claim grammatical when a citation is a subject."""
    text = re.sub(r"\[(\d+)\]\s*[–\-]\s*\[(\d+)\]", r"[\1-\2]", sentence or "")
    # Citation at the end of a clause or right after an author name -> drop it.
    text = re.sub(r"\s*" + CITATION_PATTERN.pattern + r"(?=\s*(?:[.,;:)!?]|$))", "", text)
    text = _drop_citations_after_author_names(text)
    # Remaining citations act as nouns ("[3] proposes ...", "Both [1] and [2] ...").
    text = CITATION_PATTERN.sub("the cited study", text)
    text = re.sub(r"^the cited study", "The cited study", text)
    text = re.sub(r"\*\*|__|`", "", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


# Counts words (word-like pieces, hyphens and apostrophes allowed).
def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


# One sentence of the review with its context: the text, the section heading it is in,
# the paper numbers it cites, and whether it is in the research-gaps section or the conclusion.
@dataclass
class ReviewSentence:
    text: str
    section: str
    citations: List[int]
    is_gap_section: bool
    is_conclusion: bool

    # The sentence without its citation marks (strip_citations) - this is the text that gets fact-checked.
    @property
    def claim(self) -> str:
        return strip_citations(self.text)


# True if a heading is about research gaps: contains gap, open direction, open problem, future,
# limitation or challenge.
def is_gap_heading(heading: str) -> bool:
    h = heading.lower()
    return any(k in h for k in ("gap", "open direction", "open problem", "future", "limitation", "challenge"))


# True if a heading contains 'conclusion' or 'summary'.
def is_conclusion_heading(heading: str) -> bool:
    return "conclusion" in heading.lower() or "summary" in heading.lower()


# Turns the whole review body into a list of ReviewSentence objects.
# For every section it splits sentences, skips very short ones (less than 6 words after removing citations)
# and records the section, the cited paper numbers and the gap/conclusion flags.
def review_sentences(review: str, min_words: int = 6) -> List[ReviewSentence]:
    body, _ = split_body_and_references(review)
    out: List[ReviewSentence] = []
    for heading, text in split_sections(body):
        for sentence in split_sentences(text):
            if word_count(strip_citations(sentence)) < min_words:
                continue
            out.append(
                ReviewSentence(
                    text=sentence,
                    section=heading,
                    citations=extract_citations(sentence),
                    is_gap_section=is_gap_heading(heading),
                    is_conclusion=is_conclusion_heading(heading),
                )
            )
    return out


# Reads the References section and returns {reference number: entry text}.
# Handles lines like '[3] ...', '- [3] ...' or '3. [3] ...' and joins entries that continue on the next line.
def parse_reference_entries(references_text: str) -> Dict[int, str]:
    """Map reference number -> raw entry text."""
    entries: Dict[int, str] = {}
    current: Optional[int] = None
    for line in (references_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(?:[-*]\s*)?(?:\d+\.\s*)?\[(\d+)\]\s*(.*)$", line)
        if match:
            current = int(match.group(1))
            entries[current] = match.group(2).strip()
        elif current is not None and not line.startswith("#"):
            entries[current] += " " + line
    return entries


# Pulls the paper title out of one reference line: first a quoted title (IEEE style),
# then the text after the year in brackets (APA style), otherwise the longest of the first sentences.
def parse_reference_title(entry: str) -> str:
    """Best-effort title extraction from an IEEE/APA-style reference string."""
    entry = re.sub(r"[*_]", "", entry or "").strip()
    quoted = re.search(r"[\"“”]([^\"“”]{8,300})[\"“”]", entry)
    if quoted:
        return quoted.group(1).strip().rstrip(",.")
    apa = re.search(r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)\.?\s*(.+?)(?:\.\s|\?\s|$)", entry)
    if apa:
        return apa.group(1).strip().rstrip(",.")
    parts = [p.strip() for p in re.split(r"\.\s+", entry) if p.strip()]
    if len(parts) >= 2:
        return max(parts[:3], key=len).rstrip(",.")
    return entry[:200]


# Prepares a title for comparison: lowercase, remove HTML tags and punctuation, squeeze spaces.
def normalize_title(title: str) -> str:
    title = re.sub(r"<[^>]+>", " ", str(title or "").lower())
    title = re.sub(r"[^a-z0-9 ]+", " ", title)
    return re.sub(r"\s+", " ", title).strip()


# Score from 0 to 1 for how similar two titles are.
# Uses character matching (SequenceMatcher) and word overlap (Jaccard + containment) and takes the better one.
# Used to match review references to real papers and to detect fabricated references.
def title_similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / max(1, len(ta | tb))
    containment = len(ta & tb) / max(1, min(len(ta), len(tb)))
    return max(ratio, 0.5 * jaccard + 0.5 * containment if min(len(ta), len(tb)) >= 4 else ratio)


# ----------------------------------------------------------------------
# Models (lazy singletons)
# ----------------------------------------------------------------------

# Loads the MiniLM sentence-embedding model (all-MiniLM-L6-v2). An embedding turns a sentence into numbers
# so that sentences with similar meaning are close together.
@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


# Wrapper around an NLI (natural language inference) model.
# Given (evidence, claim) it tells how likely the evidence ENTAILS (supports) the claim and how likely it CONTRADICTS it.
class NLIScorer:
    # Loads the cross-encoder model by name and finds which output index means 'entailment'
    # and which means 'contradiction'.
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=320)
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.entail_idx = next(i for i, l in id2label.items() if l.startswith("entail"))
        self.contra_idx = next(i for i, l in id2label.items() if l.startswith("contra"))

    # Scores many (evidence, claim) pairs: runs the model in batches and returns an array with
    # [probability of entailment, probability of contradiction] for each pair.
    def score(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        """Return array [n, 2] of (P(entail), P(contradiction))."""
        if not pairs:
            return np.zeros((0, 2))
        probs = self.model.predict(list(pairs), batch_size=batch_size, apply_softmax=True, show_progress_bar=False)
        probs = np.asarray(probs)
        return np.stack([probs[:, self.entail_idx], probs[:, self.contra_idx]], axis=1)


# Returns the NLI scorer for a model name, creating it once and remembering it (cache),
# because loading the model is slow.
@lru_cache(maxsize=2)
def get_nli(model_name: str) -> NLIScorer:
    return NLIScorer(model_name)


# ----------------------------------------------------------------------
# Evidence index
# ----------------------------------------------------------------------

# Converts a value (string, dict, list of strings/dicts) into a flat list of text items,
# using fields such as claim/evidence/statement/text from dict items.
def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [" ".join(str(v) for v in value.values() if v)]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if isinstance(item, dict):
                text = " ".join(str(v) for k, v in item.items() if v and k in {"claim", "evidence", "statement", "text", "finding", "limitation"})
                if not text:
                    text = " ".join(str(v) for v in item.values() if isinstance(v, str))
                if text.strip():
                    out.append(text.strip())
            elif str(item).strip():
                out.append(str(item).strip())
        return out
    return [str(value)]


# Prepares paper full text for indexing: cuts everything after a 'References/Bibliography' heading,
# joins words broken by a line-end hyphen, and squeezes whitespace.
def _clean_full_text(text: str) -> str:
    text = re.sub(r"(?is)\n\s*(references|bibliography)\s*\n.*$", "", text or "")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Makes overlapping groups of consecutive sentences (a sliding window), for example 2 sentences at a time
# moving 1 sentence at a time. Such short windows become the evidence passages.
def _windows(sentences: List[str], size: int = 3, stride: int = 2) -> List[str]:
    if len(sentences) <= size:
        return [" ".join(sentences)] if sentences else []
    return [" ".join(sentences[i: i + size]) for i in range(0, len(sentences) - size + stride, stride)]


# One piece of evidence text: the paper number it comes from, the text, and its kind
# (abstract, fulltext or analysis).
@dataclass
class Passage:
    paper: int  # 1-based paper number
    text: str
    kind: str  # abstract | fulltext | analysis


# Builds a short author phrase for a paper: 'Liu (2025)', 'Liu and Wang (2025)', 'Liu et al. (2025)'
# or 'The authors' if no authors are known.
def _author_phrase(paper: Dict[str, Any]) -> str:
    authors = paper.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in re.split(r",|;| and ", authors.strip("[]")) if a.strip()]
    names = [str(a).strip(" '\"") for a in authors if str(a).strip(" '\"")]
    last = [n.split()[-1] for n in names if n.split()]
    if not last:
        phrase = "The authors"
    elif len(last) == 1:
        phrase = last[0]
    elif len(last) == 2:
        phrase = f"{last[0]} and {last[1]}"
    else:
        phrase = f"{last[0]} et al."
    year = paper.get("year") or paper.get("published_year") or ""
    return f"{phrase} ({year})" if year else phrase


# A searchable store of evidence from the cited papers.
# When created it builds passages from each paper: abstract sentence windows, full-text sentence windows
# (first 60,000 characters, max 500), the structured analysis items (if allowed) and the paper title.
# Then it converts all passages into MiniLM embeddings so the best passages for any claim can be found quickly.
# The evaluator creates it with use_analysis=False so claims are checked ONLY against primary text.
@dataclass
class EvidenceIndex:
    papers: List[Dict[str, Any]]
    max_fulltext_chars: int = 60000
    use_analysis: bool = True  # False -> primary source text only (used by the judge)
    passages: List[Passage] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None

    # Passage text with the authors in front ('Liu et al. (2025) report: ...') so that sentences naming an author
    # can be checked correctly.
    def premise(self, passage: "Passage") -> str:
        """Passage text prefixed with who wrote it, so author-named claims can be checked."""
        return f"{self.headers[passage.paper]} report: {passage.text}"

    # Runs right after the EvidenceIndex is created: builds the author phrases, creates all passages,
    # computes their embeddings and remembers which paper each passage belongs to.
    def __post_init__(self):
        self.headers = {n: _author_phrase(p) for n, p in enumerate(self.papers, start=1)}
        for number, paper in enumerate(self.papers, start=1):
            abstract = str(paper.get("abstract") or "")
            for window in _windows(split_sentences(abstract), 2, 1):
                self.passages.append(Passage(number, window, "abstract"))

            full = _clean_full_text(str(paper.get("full_text") or ""))[: self.max_fulltext_chars]
            if full and full[:500] != abstract[:500]:
                sentences = [s for s in split_sentences(full) if 4 <= word_count(s) <= 120]
                for window in _windows(sentences, 2, 1)[:500]:
                    self.passages.append(Passage(number, window[:PASSAGE_CHARS], "fulltext"))

            analysis = (paper.get("paper_analysis") or {}) if self.use_analysis else {}
            for key in (
                "research_problem", "objectives", "methodology", "dataset", "models_and_techniques",
                "evaluation_metrics", "key_findings", "contributions", "limitations", "future_work",
                "evidence", "evidence_records",
            ):
                for item in _as_list(analysis.get(key)):
                    if word_count(item) >= 4:
                        self.passages.append(Passage(number, item[:PASSAGE_CHARS], "analysis"))

            title = str(paper.get("title") or "")
            if title:
                self.passages.append(Passage(number, f"This paper is titled: {title}.", "analysis"))

        texts = [p.text for p in self.passages]
        self.embeddings = (
            get_embedder().encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            if texts
            else np.zeros((0, 384))
        )
        self._paper_ids = np.array([p.paper for p in self.passages])

    # How many papers are in the index.
    @property
    def n_papers(self) -> int:
        return len(self.papers)

    # Finds the k passages most similar in meaning to a claim.
    # It embeds the claim, compares it with every passage embedding, optionally only within chosen papers,
    # and returns the best k.
    def top_passages(self, claim: str, papers: Optional[Iterable[int]] = None, k: int = 6) -> List[Passage]:
        if not len(self.passages):
            return []
        query = get_embedder().encode([claim], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = self.embeddings @ query
        if papers is not None:
            mask = np.isin(self._paper_ids, list(papers))
            if not mask.any():
                return []
            sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        return [self.passages[i] for i in order if np.isfinite(sims[i])]


# Result of checking one claim: entailment score, contradiction score, the best passage and its paper number.
@dataclass
class SupportResult:
    entail: float
    contradict: float
    best_passage: str
    best_paper: Optional[int]


# Measures how well papers support a claim.
#  1. Picks candidate passages: only from the cited papers (attribution) or from any paper (if no citation).
#  2. For multi-paper claims it also builds a combined premise from the best passage of each cited paper.
#  3. The NLI model scores every (passage, claim) pair.
#  4. Returns the highest entailment, the highest contradiction, and the best passage with its paper.
def claim_support(
    claim: str,
    index: EvidenceIndex,
    nli: NLIScorer,
    papers: Optional[Sequence[int]] = None,
    k: int = 4,
) -> SupportResult:
    """
    Max-entailment support for a claim.

    * papers=None   -> support from ANY source paper (corpus-level grounding)
    * papers=[i, j] -> support only from the cited papers (attribution)
    For multi-paper claims a combined premise (best passage of each cited
    paper) is also scored, since comparisons need evidence from both sides.
    """
    if papers is not None:
        papers = [p for p in papers if 1 <= p <= index.n_papers]
        if not papers:
            return SupportResult(0.0, 0.0, "", None)

    candidates: List[Passage] = []
    if papers and len(papers) > 1:
        per_paper_k = max(2, k // len(papers) + 1)
        for p in papers:
            candidates.extend(index.top_passages(claim, [p], per_paper_k))
    else:
        candidates = index.top_passages(claim, papers, k)
    if not candidates:
        return SupportResult(0.0, 0.0, "", None)

    premises = [index.premise(c) for c in candidates]
    owners = [c.paper for c in candidates]
    if papers and len(papers) > 1:
        combined = []
        for p in papers:
            best = next((index.premise(c) for c in candidates if c.paper == p), "")
            if best:
                combined.append(best)
        if len(combined) > 1:
            premises.append(" ".join(combined))
            owners.append(None)

    scores = nli.score([(premise, claim) for premise in premises])
    best = int(np.argmax(scores[:, 0]))
    contradiction = float(scores[:, 1].max())
    return SupportResult(
        entail=float(scores[best, 0]),
        contradict=contradiction,
        best_passage=premises[best],
        best_paper=owners[best],
    )
