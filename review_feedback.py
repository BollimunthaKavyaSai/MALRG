"""
Feedback agents: Critic -> Fact-Checker -> Reviser loop for the written review.

Round r:
  1. Citation Auditor (deterministic): invalid citation numbers, papers never
     cited, uncited factual sentences, uncited gap statements, over-claiming
     wording, too few cross-paper comparison sentences.
  2. Fact-Checker (NLI pre-screen + LLM): cited claims are compared with the
     cited papers' evidence; low-entailment claims go to an LLM fact-checker.
  3. Reviser (writer LLM): fixes only the flagged sentences using the shown
     evidence, keeping everything else unchanged.
  4. The revision is ACCEPTED only if the critic score improves; otherwise the
     loop stops and the previous draft is kept (monotone improvement).

The same machinery powers human-in-the-loop feedback: a user's comments are
added as an extra issue and the Reviser applies them together with the
automatic findings.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from claim_checking import build_claim_items, judge_claims
from grounding import (
    CERTAINTY_PHRASES,
    COMPARATIVE_MARKERS,
    FEEDBACK_NLI_MODEL,
    EvidenceIndex,
    claim_support,
    get_nli,
    review_sentences,
    split_body_and_references,
    word_count,
)
from llm_client import estimate_tokens, invoke_text, make_llm

FACT_CHECK_MODEL = os.environ.get("MALRG_FACTCHECK_MODEL", "openai/gpt-oss-20b")
REVISER_MODEL = os.environ.get("MALRG_WRITER_MODEL", "openai/gpt-oss-120b")
NLI_AUTO_ACCEPT = 0.85       # cited claims above this entailment skip the LLM check
MAX_LLM_CHECKS = int(os.environ.get("MALRG_FEEDBACK_MAX_CHECKS", "12"))
MAX_ISSUES_IN_PROMPT = 14
REVISER_PROMPT_BUDGET = int(os.environ.get("MALRG_WRITER_PROMPT_BUDGET", "4800"))

FACTUAL_VERBS = re.compile(
    r"\b(propos|introduc|achiev|report|show|demonstrat|evaluat|employ|use[sd]?\b|using|"
    r"train|outperform|improv|reduc|increas|found|observ|measur|develop|present|"
    r"compar|apply|applie|adopt|leverag|exhibit|attain|obtain|reach)",
    re.I,
)


# Result of one critic pass over a review: a score (0-100), the list of issues found, and some statistics.
@dataclass
class FeedbackReport:
    score: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    # Counts the issues by type (for example 2 UNSUPPORTED, 1 UNCITED_PAPER) for logging and display.
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for issue in self.issues:
            out[issue["type"]] = out.get(issue["type"], 0) + 1
        return out


# Makes the numbered paper list shown to the Reviser, one line per paper: [1] "Title" (year).
def _paper_directory(papers: List[Dict[str, Any]]) -> str:
    lines = []
    for number, paper in enumerate(papers, start=1):
        lines.append(f"[{number}] \"{str(paper.get('title', '')).strip()}\" ({paper.get('year') or paper.get('published_year') or 'n.d.'})")
    return "\n".join(lines)


# The CRITIC: checks a review and returns a score plus a list of problems.
#  1. Citation numbers that do not exist ([7] when only 5 papers) -> INVALID_CITATION.
#  2. Papers never cited in the text -> UNCITED_PAPER.
#  3. Factual-looking sentences with no citation -> UNCITED_CLAIM / GAP_UNCITED.
#  4. Over-certain words (proves, conclusively ...) -> OVERCLAIM.
#  5. Too few sentences that compare 2+ papers -> LOW_SYNTHESIS; no gaps section -> MISSING_GAP_SECTION.
#  6. FACT-CHECK: each cited sentence is compared with its cited papers using an NLI model; sentences with
#     entailment below 0.85 (up to 12 of the weakest) go to an LLM fact-checker -> UNSUPPORTED / PARTIAL / CONTRADICTED.
# Score = 40% supported claims + 15% paper coverage + 15% citation validity + 10% cited facts + 15% synthesis
# + 5% calibrated wording (all x100).
def critique_review(
    review: str,
    papers: List[Dict[str, Any]],
    index: Optional[EvidenceIndex] = None,
    use_llm: bool = True,
) -> FeedbackReport:
    index = index or EvidenceIndex(papers)
    n_papers = len(papers)
    sentences = review_sentences(review)
    issues: List[Dict[str, Any]] = []

    content = [s for s in sentences if s.section and not s.is_conclusion]
    cited = [s for s in sentences if s.citations]
    markers = [c for s in sentences for c in s.citations]
    valid_markers = [c for c in markers if 1 <= c <= n_papers]

    # 1. Invalid citation numbers ---------------------------------------
    for s in sentences:
        bad = [c for c in s.citations if not 1 <= c <= n_papers]
        if bad:
            issues.append({
                "type": "INVALID_CITATION",
                "sentence": s.text,
                "problem": f"cites {bad} but only [1]-[{n_papers}] exist",
            })

    # 2. Papers never cited ---------------------------------------------
    cited_set = set(valid_markers)
    for paper in range(1, n_papers + 1):
        if paper not in cited_set:
            notes = [p.text for p in index.passages if p.paper == paper and p.kind == "analysis"][:3]
            issues.append({
                "type": "UNCITED_PAPER",
                "paper": paper,
                "problem": f"paper [{paper}] is never cited in the body",
                "evidence": [index.premise(p) for p in index.passages if p.paper == paper and p.kind == "analysis"][:3] or notes,
            })

    # 3. Uncited factual sentences --------------------------------------
    uncited_factual = [
        s for s in content
        if not s.citations and (re.search(r"\d", s.claim) or FACTUAL_VERBS.search(s.claim))
    ]
    for s in uncited_factual:
        issues.append({
            "type": "GAP_UNCITED" if s.is_gap_section else "UNCITED_CLAIM",
            "sentence": s.text,
            "problem": "factual statement without a citation",
        })

    # 4. Over-claiming wording ------------------------------------------
    overclaims = []
    for s in sentences:
        lowered = s.text.lower()
        hits = [p for p in CERTAINTY_PHRASES if p in lowered]
        if hits:
            overclaims.append(s)
            issues.append({"type": "OVERCLAIM", "sentence": s.text, "problem": f"over-certain wording: {hits}"})

    # 5. Cross-paper synthesis density ----------------------------------
    multi = [s for s in sentences if len({c for c in s.citations if 1 <= c <= n_papers}) >= 2]
    target_multi = max(2, math.ceil(n_papers / 2))
    if n_papers >= 2 and len(multi) < target_multi:
        issues.append({
            "type": "LOW_SYNTHESIS",
            "problem": (
                f"only {len(multi)} sentences compare two or more papers; add at least "
                f"{target_multi - len(multi)} explicit comparison sentences that cite every compared paper"
            ),
        })
    if not any(s.is_gap_section for s in sentences):
        issues.append({"type": "MISSING_GAP_SECTION", "problem": "no 'Research Gaps and Open Directions' section"})

    # 6. Fact-check cited claims ----------------------------------------
    checked = 0
    flagged_unsupported = 0.0
    nli = get_nli(FEEDBACK_NLI_MODEL)
    valid_cited = [s for s in cited if any(1 <= c <= n_papers for c in s.citations)]
    scored = []
    for s in valid_cited:
        support = claim_support(s.claim, index, nli, s.citations)
        scored.append((support.entail, s))
    to_check = [s for entail, s in sorted(scored, key=lambda t: t[0]) if entail < NLI_AUTO_ACCEPT][:MAX_LLM_CHECKS]

    if use_llm and to_check:
        items, pool = build_claim_items(to_check, index)
        verdicts = judge_claims(items, pool, FACT_CHECK_MODEL, label="Fact-Checker Agent")
        for item in items:
            verdict = verdicts.get(item.id, {}).get("label", "UNJUDGED")
            checked += 1
            if verdict in {"UNSUPPORTED", "CONTRADICTED", "PARTIAL"}:
                flagged_unsupported += 0.5 if verdict == "PARTIAL" else 1.0
                issues.append({
                    "type": verdict,
                    "sentence": item.sentence,
                    "problem": verdicts[item.id].get("reason", ""),
                    "evidence": [pool[e][1] for e in item.evidence_ids][:3],
                })

    supported_fraction = 1.0 - (flagged_unsupported / max(1, len(valid_cited)))
    coverage = len(cited_set) / max(1, n_papers)
    validity = len(valid_markers) / max(1, len(markers)) if markers else 0.0
    uncited_rate = len(uncited_factual) / max(1, len(content))
    synthesis = min(1.0, len(multi) / target_multi) if n_papers >= 2 else 1.0
    overclaim = min(1.0, len(overclaims) / 3)

    score = 100 * (
        0.40 * supported_fraction
        + 0.15 * coverage
        + 0.15 * validity
        + 0.10 * (1 - uncited_rate)
        + 0.15 * synthesis
        + 0.05 * (1 - overclaim)
    )
    stats = {
        "sentences": len(sentences),
        "cited_sentences": len(cited),
        "llm_checked": checked,
        "supported_fraction": round(supported_fraction, 4),
        "paper_coverage": round(coverage, 4),
        "citation_validity": round(validity, 4),
        "uncited_factual_rate": round(uncited_rate, 4),
        "multi_citation_sentences": len(multi),
        "overclaims": len(overclaims),
        "comparative_markers": sum(
            1 for s in multi for m in COMPARATIVE_MARKERS if m in s.text.lower()
        ),
        "word_count": word_count(split_body_and_references(review)[0]),
    }
    return FeedbackReport(score=round(score, 2), issues=issues, stats=stats)


REVISER_PROMPT = """You are the Reviser agent of a multi-agent literature-review system.
The Citation Auditor and the Fact-Checker reviewed the DRAFT below and found ISSUES.
Produce a corrected version of the draft.

RULES:
1. Fix every listed issue and leave every other sentence unchanged.
2. UNSUPPORTED / PARTIAL: rewrite the sentence so it states only what the shown evidence
   supports (keep the citation), or delete it if the evidence supports nothing in it.
3. CONTRADICTED: correct the sentence according to the evidence.
4. UNCITED_CLAIM / GAP_UNCITED: add the correct citation [i] from the PAPER DIRECTORY when the
   evidence supports the statement; otherwise rephrase it as a cautious framing statement.
5. UNCITED_PAPER: add one or two accurate sentences about that paper, based only on its evidence,
   in the most relevant thematic section.
6. LOW_SYNTHESIS: add explicit comparison sentences ("[a] ..., whereas [b] ...") that cite every
   compared paper and use only facts already present in the draft or the evidence.
7. OVERCLAIM: replace over-certain wording with calibrated wording.
8. USER_FEEDBACK: apply the reviewer's request as long as it does not require unsupported facts.
9. Never invent numbers, datasets, methods or findings. Never add papers outside the directory.
10. Keep all headings. Do NOT output a References section (it is appended automatically).

PAPER DIRECTORY:
{directory}

ISSUES:
{issues}

DRAFT (without references):
{draft}

Return ONLY the full revised review in Markdown, starting with its first heading."""


# Turns the issue list into text for the Reviser prompt.
# Issues are sorted by importance (user feedback first, then contradictions, invalid citations, unsupported ...),
# cut to a limited number, and each is shown with its sentence and a little evidence.
def _format_issues(issues: List[Dict[str, Any]], limit: int = MAX_ISSUES_IN_PROMPT, evidence_per_issue: int = 2) -> str:
    priority = {
        "USER_FEEDBACK": 0, "CONTRADICTED": 1, "INVALID_CITATION": 2, "UNSUPPORTED": 3,
        "UNCITED_PAPER": 4, "PARTIAL": 5, "GAP_UNCITED": 6, "UNCITED_CLAIM": 7,
        "LOW_SYNTHESIS": 8, "MISSING_GAP_SECTION": 9, "OVERCLAIM": 10,
    }
    ordered = sorted(issues, key=lambda i: priority.get(i["type"], 99))[:limit]
    lines = []
    for n, issue in enumerate(ordered, start=1):
        line = f"{n}. [{issue['type']}] {issue.get('problem', '')}"
        if issue.get("sentence"):
            line += f"\n   Sentence: \"{issue['sentence']}\""
        for evidence in issue.get("evidence", [])[:evidence_per_issue]:
            line += f"\n   Evidence: {evidence[:420]}"
        lines.append(line)
    return "\n".join(lines)


# The REVISER: rewrites the flagged parts of the review.
#  1. Builds a prompt with the paper directory, the issues and the draft (without References).
#     If the prompt is too big, it shows fewer issues/evidence until it fits the token budget.
#  2. The rules tell it to fix only the listed issues, never invent numbers or papers, keep headings,
#     and apply any USER_FEEDBACK as long as no unsupported facts are needed.
#  3. Calls gpt-oss-120b (temperature 0.1), removes any References/code fences it adds,
#     and postprocess() re-attaches the clean reference list.
def revise_review(
    review: str,
    issues: List[Dict[str, Any]],
    papers: List[Dict[str, Any]],
    postprocess: Callable[[str], str],
) -> str:
    body, _ = split_body_and_references(review)
    # Fit the per-minute token budget: fewer issues / less evidence if needed.
    for limit, per_issue in ((MAX_ISSUES_IN_PROMPT, 2), (10, 2), (10, 1), (7, 1), (5, 1), (4, 0)):
        prompt = REVISER_PROMPT.format(
            directory=_paper_directory(papers),
            issues=_format_issues(issues, limit, per_issue),
            draft=body.strip(),
        )
        if estimate_tokens(prompt) <= REVISER_PROMPT_BUDGET:
            break
    print(f"   📏 [Reviser Agent] prompt ≈ {estimate_tokens(prompt)} tokens")
    llm = make_llm(REVISER_MODEL, temperature=0.1, max_tokens=int(os.environ.get("MALRG_WRITER_MAX_TOKENS", "8000")), reasoning_effort="medium")
    revised = invoke_text(llm, prompt, label="Reviser Agent").strip()
    revised = re.sub(r"(?is)\n#{1,6}\s*references\s*\n.*$", "", revised).strip()
    revised = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", revised).strip()
    return postprocess(revised)


# Automatic improve-the-review loop (used by agent 10 in mode M5).
#  1. Critic scores the draft (round 0).
#  2. For each round (default 2): the Reviser rewrites -> the Critic scores the new version.
#  3. The new version is ACCEPTED only if the score rises by at least 0.5 and the text did not shrink
#     below 60% of its length. Otherwise the loop stops and the previous draft is kept.
# Returns the final review, a log of every round and the final report.
def run_feedback_loop(
    review: str,
    papers: List[Dict[str, Any]],
    postprocess: Callable[[str], str],
    max_rounds: int = 2,
    min_gain: float = 0.5,
) -> Dict[str, Any]:
    """Iteratively critique and revise; keep only revisions that raise the critic score."""
    index = EvidenceIndex(papers)
    current = review
    report = critique_review(current, papers, index)
    log = [{"round": 0, "score": report.score, "issue_counts": report.counts(), "stats": report.stats, "accepted": True}]
    print(f"   🧪 [Feedback Loop]: draft critic score = {report.score} | issues = {report.counts()}")

    for round_no in range(1, max_rounds + 1):
        actionable = [i for i in report.issues if i["type"] != "OVERCLAIM" or len(report.issues) < 3]
        if not actionable:
            print("   ✅ [Feedback Loop]: no actionable issues left.")
            break
        candidate = revise_review(current, report.issues, papers, postprocess)
        candidate_report = critique_review(candidate, papers, index)

        shrunk = word_count(candidate) < 0.6 * word_count(current)
        accepted = candidate_report.score >= report.score + min_gain and not shrunk
        log.append({
            "round": round_no,
            "score": candidate_report.score,
            "issue_counts": candidate_report.counts(),
            "stats": candidate_report.stats,
            "accepted": accepted,
            "rejected_reason": None if accepted else ("revision shrank the review" if shrunk else "no critic-score gain"),
        })
        print(
            f"   🔁 [Feedback Loop]: round {round_no} score {report.score} -> {candidate_report.score} "
            f"({'accepted' if accepted else 'rejected'})"
        )
        if not accepted:
            break
        current, report = candidate, candidate_report

    return {"final_review_md": current, "feedback_log": log, "final_report": asdict(report)}


# Human-in-the-loop revision (the feedback box on the Manuscript tab).
#  1. Critic scores the current review (before).
#  2. The user's comment is added as the top-priority issue USER_FEEDBACK in front of the automatic issues.
#  3. One Reviser pass rewrites the review.
#  4. Critic scores the new review (after). Returns both versions and both reports.
# Note: the new review is not rejected when the score drops; the caller simply stores it.
def revise_with_user_feedback(
    review: str,
    comments: str,
    papers: List[Dict[str, Any]],
    postprocess: Callable[[str], str],
) -> Dict[str, Any]:
    """Human-in-the-loop revision: user comments + automatic findings in one Reviser pass."""
    index = EvidenceIndex(papers)
    before = critique_review(review, papers, index)
    issues = [{"type": "USER_FEEDBACK", "problem": comments.strip()}] + before.issues
    revised = revise_review(review, issues, papers, postprocess)
    after = critique_review(revised, papers, index)
    return {
        "final_review_md": revised,
        "previous_review_md": review,
        "critic_before": asdict(before),
        "critic_after": asdict(after),
    }
