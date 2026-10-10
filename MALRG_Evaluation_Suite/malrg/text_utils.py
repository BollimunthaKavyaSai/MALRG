"""Text helpers: sentence splitting, citation parsing, chunking, quote matching."""

import re
from difflib import SequenceMatcher

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,–\-]\s*\d+)*)\]")
_ABBREV = ("et al.", "e.g.", "i.e.", "vs.", "Fig.", "fig.", "Eq.", "approx.", "cf.", "No.", "al.")


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (title or "").lower()).strip()


def split_sentences(text: str):
    """Sentence splitter that keeps citations attached and respects abbreviations."""
    text = normalize_ws(text)
    if not text:
        return []
    protected = text
    for i, ab in enumerate(_ABBREV):
        protected = protected.replace(ab, ab.replace(".", f"<DOT{i}>"))
    protected = re.sub(r"(\d)\.(\d)", r"\1<DEC>\2", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[(\"'])", protected)
    out = []
    for p in parts:
        for i, ab in enumerate(_ABBREV):
            p = p.replace(f"<DOT{i}>", ".")
        p = p.replace("<DEC>", ".").strip()
        if p:
            out.append(p)
    return out


def parse_citations(sentence: str):
    """Return the sorted list of integer citation ids in a sentence ([1], [2, 3], [1-3])."""
    ids = set()
    for group in CITATION_RE.findall(sentence):
        for part in re.split(r"\s*,\s*", group):
            rng = re.split(r"\s*[–\-]\s*", part)
            try:
                if len(rng) == 2:
                    a, b = int(rng[0]), int(rng[1])
                    if 0 < b - a < 20:
                        ids.update(range(a, b + 1))
                        continue
                ids.add(int(rng[0]))
            except ValueError:
                continue
    return sorted(ids)


def strip_citations(sentence: str) -> str:
    return normalize_ws(CITATION_RE.sub("", sentence))


def split_markdown_sections(md: str):
    """Split a markdown document into [(heading, body)] keeping order."""
    sections = []
    current, buf = "Preamble", []
    for line in md.splitlines():
        m = re.match(r"^\s{0,3}(#{1,4})\s+(.*)$", line)
        if m:
            if buf:
                sections.append((current, "\n".join(buf).strip()))
            current, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((current, "\n".join(buf).strip()))
    return sections


def is_reference_heading(heading: str) -> bool:
    return bool(re.search(r"reference|bibliograph|works cited", heading, re.I))


def is_gap_heading(heading: str) -> bool:
    return bool(re.search(r"gap|open (problem|challenge|question)|future (direction|work)|limitation", heading, re.I))


def body_sentences(md: str):
    """[(section_heading, sentence)] for all prose (excluding the reference list)."""
    out = []
    for heading, body in split_markdown_sections(md):
        if is_reference_heading(heading):
            continue
        for para in re.split(r"\n\s*\n", body):
            lines = []
            for line in para.splitlines():
                line = line.strip()
                if not line or line.startswith("|") or re.match(r"^[-*_]{3,}$", line):
                    continue
                line = re.sub(r"^[-*+]\s+|^\d+\.\s+", "", line)
                line = re.sub(r"[*`]+", "", line)
                lines.append(line)
            for s in split_sentences(" ".join(lines)):
                if len(s.split()) >= 5:
                    out.append((heading, s))
    return out


def reference_section(md: str) -> str:
    for heading, body in split_markdown_sections(md):
        if is_reference_heading(heading):
            return body
    return ""


def chunk_text(text: str, words: int, overlap: int):
    toks = text.split()
    if not toks:
        return []
    step = max(1, words - overlap)
    chunks = []
    for start in range(0, len(toks), step):
        piece = toks[start:start + words]
        if len(piece) < 25 and chunks:
            break
        chunks.append(" ".join(piece))
        if start + words >= len(toks):
            break
    return chunks


def quote_in_text(quote: str, text_norm: str, threshold: float = 0.85) -> bool:
    """True if `quote` appears (near-)verbatim in the normalized source text."""
    q = re.sub(r"[^a-z0-9 ]", " ", (quote or "").lower())
    q = normalize_ws(q)
    if len(q.split()) < 4:
        return False
    if q in text_norm:
        return True
    # Sliding fuzzy match over windows that share the quote's first token.
    words = q.split()
    n = len(words)
    first = words[0]
    tw = text_norm.split()
    best = 0.0
    for i, w in enumerate(tw):
        if w != first and (i % 7):
            continue
        window = " ".join(tw[i:i + n + 3])
        r = SequenceMatcher(None, q, window).ratio()
        if r > best:
            best = r
            if best >= threshold:
                return True
    return best >= threshold


def norm_for_match(text: str) -> str:
    return normalize_ws(re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()))
