import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import os
import logging

logger = logging.getLogger(__name__)
import urllib.request
import urllib.parse
import arxiv
import uuid
import json
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from online_search import fetch_and_store
from query_understanding import understand_research_query
from paper_extraction import extract_papers_content
from paper_analysis import analyze_papers
from cross_paper_analysis import analyze_cross_papers
from llm_client import (
    PROMPT_TOKEN_BUDGET,
    estimate_tokens,
    snapshot_token_usage,
    DailyQuotaExhausted,
    LLMUnavailable,
    get_groq_api_key,
    invoke_text,
    make_llm,
    strict_mode,
)
from review_feedback import run_feedback_loop, revise_with_user_feedback

# =====================================================================
# SYSTEM CONFIGURATION & ENVIRONMENT SETUP
# =====================================================================
# The Groq API key is read only from the environment / .env file.
load_dotenv()
GROQ_API_KEY = get_groq_api_key()

# ---------------------------------------------------------------------
# ABLATION CONFIGURATIONS
# ---------------------------------------------------------------------
# M0 (vanilla single prompt) does not use the graph; see evaluation/ablation.py.
ABLATION_MODES = {
    "M1_retrieval_only": "Retrieval + writer (abstracts only; no analysis, cross-paper, gap agents)",
    "M2_no_cross_paper": "Full pipeline without the cross-paper synthesis agent",
    "M3_no_skeptic": "Full pipeline without the Skeptic/Verifier (proposer gaps accepted)",
    "M4_full": "Complete multi-agent pipeline (no feedback loop)",
    "M5_full_feedback": "Complete pipeline + critic feedback/revision loop (proposed)",
}
DEFAULT_ABLATION_MODE = os.environ.get("MALRG_DEFAULT_MODE", "M5_full_feedback")
# Search Agent: if MongoDB lacks enough relevant papers, fetch the rest from OpenAlex (set 0 to disable).
ONLINE_SEARCH_ENABLED = os.environ.get("MALRG_ONLINE_SEARCH", "1") != "0"
FEEDBACK_MAX_ROUNDS = int(os.environ.get("MALRG_FEEDBACK_ROUNDS", "2"))

# Initialize FastAPI App
app = FastAPI(title="Multi-Agent Literature Review Generator API")

# Enable CORS (Cross-Origin Resource Sharing) so your React frontend can safely talk to it later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory simulation tracker for managing asynchronous task memory states
# In production, this will be handled by a PostgreSQL database database mapping layer
TASKS_DB: Dict[str, Dict[str, Any]] = {}

# Health check: a tiny endpoint to see that the Python engine is running.
# Open http://127.0.0.1:8000/health in a browser and it answers with a small JSON:
#   status 'ok', the service name, the current time, and how many review tasks are held in memory (TASKS_DB).
# The Node server and the start scripts use this kind of check to know the AI engine is alive.
@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "Nexus-Reviews AI Python Engine",
        "timestamp": datetime.utcnow().isoformat(),
        "tasks_in_memory": len(TASKS_DB),
    }

# =====================================================================
# STATE MATRIX & SCHEMAS
# =====================================================================
# ResearchState = the shared notebook of ONE review run.
# LangGraph passes this one object from agent to agent. Each agent reads what earlier agents wrote
# and fills in its own part, so nothing has to be passed around by hand.
# Fields in the order they get filled:
#   topic, start_year, end_year, max_papers ...... what the user typed in the form
#   ablation_mode, feedback_rounds ............... which version (M1..M5) of the pipeline to run
#   raw_papers_list .............................. papers chosen by the Search agent
#   extracted_papers ............................. same papers with their full text (PDF extraction)
#   analyzed_papers .............................. per-paper evidence: method, results, limitations
#   cross_paper_analysis ......................... themes, agreements, contradictions across papers
#   research_gap_analysis / verified_research_gaps  proposed, challenged and verified research gaps
#   summarized_papers, taxonomy_map, citations_list  helper data for the writer
#   final_review_md / draft_review_md ............ the written literature review (Markdown)
#   feedback_log ................................. what the critic/reviser loop did
# The shared clipboard: topic, years, papers, analysis, gaps, review ...
class ResearchState(BaseModel):
    topic: str
    start_year: int = 2020
    end_year: int = 2025
    max_papers: int = 2

    # Ablation switch (see ABLATION_MODES) and feedback-loop trace
    ablation_mode: str = "M5_full_feedback"
    feedback_rounds: int = 2
    feedback_log: List[Dict[str, Any]] = []
    draft_review_md: str = ""

    # Section 1
    raw_papers_list: List[Dict[str, Any]] = []
    errors: List[str] = []

    # Section 2
    extracted_papers: List[Dict[str, Any]] = []
    analyzed_papers: List[Dict[str, Any]] = []

    # Section 3
    cross_paper_analysis: Dict[str, Any] = {}

    # Evidence-grounded research-gap discovery and verification
    research_gap_analysis: Dict[str, Any] = {}
    verified_research_gaps: List[Dict[str, Any]] = []

    # Existing downstream components
    summarized_papers: List[Dict[str, Any]] = []
    taxonomy_map: Dict[str, List[str]] = {}
    citations_list: List[Dict[str, Any]] = []
    final_review_md: str = ""

# Shape of a one-paper summary that an LLM is asked to return:
# problem addressed, methodology, dataset used, results, limitations and future work.
# It is a Pydantic model: the field descriptions tell the LLM what each field must contain,
# and Pydantic checks that the answer really has those fields.
class PaperSummarySchema(BaseModel):
    problem_addressed: str = Field(description="The primary research problem that this paper aims to solve.")
    methodology: str = Field(description="The technical methods, algorithms, or frameworks used.")
    dataset_used: str = Field(description="The data sets or testing parameters used.")
    results: str = Field(description="The quantitative or qualitative outcomes discovered.")
    limitations: str = Field(description="Any weaknesses or constraints identified.")
    future_work: str = Field(description="What the authors suggest as next steps.")

# One 'theme' of the literature, for example 'Retrieval strategies'.
# It holds the theme name and the exact titles of the papers that belong to that theme.
# The taxonomy agent returns a list of these.
class TaxonomyCategory(BaseModel):
    category_name: str = Field(description="A meaningful academic classification title.")
    paper_titles: List[str] = Field(description="The exact titles of matching papers.")

# The full answer shape for the taxonomy agent: a list of TaxonomyCategory items.
# Used with with_structured_output so the LLM must answer in exactly this layout.
class TaxonomySchema(BaseModel):
    categories: List[TaxonomyCategory] = Field(description="A breakdown list of all discovered academic themes.")

# Shape of one formatted reference: an IEEE style string and an APA 7 style string.
# (The citation agent currently builds these strings with plain Python code, not with an LLM.)
class CitationSchema(BaseModel):
    ieee_format: str = Field(description="The formal IEEE format string.")
    apa_format: str = Field(description="The formal APA 7th edition format string.")


# One research gap proposed by the Proposer agent.
# gap_statement = the gap in one sentence, rationale = why the evidence says it is unresolved,
# supporting_papers = exact titles of papers that back it, supporting_evidence = short evidence statements.
class ResearchGapCandidate(BaseModel):
    gap_statement: str = Field(description="A specific, evidence-grounded research gap emerging from the supplied papers.")
    rationale: str = Field(description="Why the supplied evidence indicates that this is unresolved.")
    supporting_papers: List[str] = Field(description="Exact titles of supporting papers.")
    supporting_evidence: List[str] = Field(description="Evidence statements grounded only in supplied analysis.")


# The answer shape of the Proposer: a list of ResearchGapCandidate items.
class ResearchGapProposalSchema(BaseModel):
    candidates: List[ResearchGapCandidate] = Field(description="Candidate research gaps derived only from supplied evidence.")


# The Skeptic's opinion about ONE candidate gap.
# challenge_valid = does the gap survive, unsupported_claims / concerns = what is weak,
# required_evidence = what would be needed to prove it, revised_gap_statement = a more careful wording.
class ResearchGapChallenge(BaseModel):
    gap_statement: str = Field(description="The candidate gap being challenged.")
    challenge_valid: bool = Field(description="Whether the candidate survives skeptical scrutiny.")
    unsupported_claims: List[str] = Field(description="Unsupported or overstated claims.")
    concerns: List[str] = Field(description="Specific weaknesses or evidence limitations.")
    required_evidence: List[str] = Field(description="Evidence needed to verify the gap.")
    revised_gap_statement: str = Field(description="A cautious revised version of the gap.")


# The answer shape of the Skeptic: one ResearchGapChallenge for each candidate gap.
class ResearchGapChallengeSchema(BaseModel):
    challenges: List[ResearchGapChallenge] = Field(description="Skeptical assessment of the proposed gaps.")


# One gap that the Verifier accepted.
# It keeps the gap sentence, supporting paper titles, the evidence, the reason it passed,
# and verified=True (only set when the gap is really supported).
class VerifiedResearchGap(BaseModel):
    gap_statement: str = Field(description="An evidence-supported research gap.")
    supporting_papers: List[str] = Field(description="Exact titles of supporting papers.")
    evidence_basis: List[str] = Field(description="Evidence supporting the verified gap.")
    verification_reason: str = Field(description="Why the gap passed verification.")
    verified: bool = Field(description="True only when the gap is sufficiently supported.")


# The answer shape of the Verifier: a list of VerifiedResearchGap items.
class ResearchGapVerificationSchema(BaseModel):
    verified_gaps: List[VerifiedResearchGap] = Field(description="Only genuinely verified research gaps.")


# =====================================================================
# UPDATE REQUEST SCHEMAS IN APP.PY
# =====================================================================
# The data a client must send to start a new review (this is what /api/review/generate accepts).
# topic (text), start_year and end_year (1900-2100), max_papers (1 to 20), and the ablation mode
# (default M5 = full system with feedback loop). FastAPI checks these automatically,
# so a wrong value such as max_papers=50 is rejected before any agent runs.
# The order form: topic, years, paper count. It rejects wrong values (for example 50 papers).

class GenerationRequest(BaseModel):
    topic: str
    start_year: int = Field(default=2020, ge=1900, le=2100)
    end_year: int = Field(default=2025, ge=1900, le=2100)
    max_papers: int = Field(
        default=2, 
        ge=1,  # Greater than or equal to 1
        le=20, # Less than or equal to 20 (Hard Cap Rule)
        description="The number of research papers to pull from the repository."
    )
    ablation_mode: str = Field(
        default=DEFAULT_ABLATION_MODE,
        description="One of ABLATION_MODES; the proposed system is M5_full_feedback.",
    )


# The data sent when a human reviewer gives feedback on a finished review.
# Only one field, 'comments', which must be between 3 and 4000 characters.
# The feedback form: only a comment of 3 to 4000 characters.
class UserFeedbackRequest(BaseModel):
    comments: str = Field(min_length=3, max_length=4000)

import urllib.parse

# =====================================================================
# HARDENED TARGETED ACADEMIC SEARCH QUERY FORMATTER
# =====================================================================
# Cleans a list of values into a list of unique, non-empty strings.
# Steps: turn each value into text, squeeze repeated spaces, skip empty ones, and skip
# duplicates (ignoring upper/lower case) while keeping the original order.
# Used for search queries and concept lists so the same phrase is not used twice.
# Removes empty and repeated text from a list.
def _unique_strings(values: List[Any]) -> List[str]:
    """Return cleaned, case-insensitively unique strings while preserving order."""
    result = []
    seen = set()
    for value in values or []:
        item = re.sub(r"\s+", " ", str(value or "")).strip()
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# Builds the 'must-match' keyword groups used by the topic relevance gate.
# It joins the user's query, the research intent, core concepts, subtopics and related concepts into one text.
# Then it checks five ideas - multi-agent, review task, retrieval, generation, citation -
# and for every idea that appears in the query it adds a group of related words.
# Example: for 'multi-agent literature review' it creates the groups multi_agent and review_task,
# so a later paper must mention both ideas to count as on-topic.
# Looks at your topic and makes lists of “must-have ideas” (for example multi-agent, review task).
def _query_anchor_groups(original_query: str, research_intent: str, core_concepts: List[Any],
                         subtopics: List[Any], related_concepts: List[Any]) -> Dict[str, List[str]]:
    """Create deterministic concept groups used to prevent semantically adjacent retrieval."""
    text = " ".join([
        original_query,
        research_intent,
        " ".join(_unique_strings(core_concepts)),
        " ".join(_unique_strings(subtopics)),
        " ".join(_unique_strings(related_concepts)),
    ]).lower()

    groups = {}

    multi_agent_terms = [
        "multi-agent", "multi agent", "multiagent", "multiple agents",
        "agent collaboration", "agent coordination", "agentic ai", "ai agents",
        "llm agents", "agent-based", "agent based",
    ]
    review_task_terms = [
        "literature review", "literature-review", "review generation",
        "review generation", "automated literature review", "automatic literature review",
        "systematic review", "scholarly review", "scientific literature review",
        "literature synthesis", "scientific synthesis", "scholarly synthesis",
        "review automation", "research synthesis",
    ]
    retrieval_terms = [
        "information retrieval", "scholarly retrieval", "document retrieval",
        "paper retrieval", "academic search", "scholarly search", "citation retrieval",
        "retrieval augmented", "retrieval-augmented",
    ]
    generation_terms = [
        "text generation", "language generation", "llm", "large language model",
        "generative ai", "generation", "summarization", "synthesis",
    ]
    citation_terms = [
        "citation", "reference", "bibliographic", "citation grounded", "citation-grounded",
    ]

    if any(term in text for term in multi_agent_terms):
        groups["multi_agent"] = multi_agent_terms
    if any(term in text for term in review_task_terms):
        groups["review_task"] = review_task_terms
    if any(term in text for term in retrieval_terms):
        groups["retrieval"] = retrieval_terms
    if any(term in text for term in generation_terms):
        groups["generation"] = generation_terms
    if any(term in text for term in citation_terms):
        groups["citation"] = citation_terms

    return groups


# Makes one lowercase text from a paper (title + abstract + venue).
# The relevance gate searches this text for the topic keywords.
# Joins a paper’s title, abstract and venue into one text.

def _paper_text_for_relevance(paper: Dict[str, Any]) -> str:
    """Build the text used by the deterministic topic-relevance gate."""
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    venue = str(paper.get("venue") or "")
    return re.sub(r"\s+", " ", f"{title}. {abstract}. {venue}").strip().lower()


# Returns True if at least one of the given words/phrases appears inside the text (ignoring case).
# Does the text contain any of these words?
def _contains_any_term(text: str, terms: List[str]) -> bool:
    return any(term.lower() in text for term in terms)


_CONCEPT_STOPWORDS = {"a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "using", "based", "via",
                      "by", "from", "into", "towards", "toward", "its", "their", "approach", "approaches",
                      "method", "methods", "technique", "techniques", "study", "studies", "system", "systems"}


# A very simple 'word root' maker so that network and networks, or forecast and forecasting, count as the same word.
# It removes one common ending (ing, ies, es, s) but only when at least 4 letters remain.
# Cuts word endings so “network” and “networks” match.
def _stem(word: str) -> str:
    """Very light stemming so that network/networks or forecast/forecasting match."""
    for suffix in ("ing", "ies", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


# Splits a concept like 'graph neural networks' into its meaningful words.
# Words such as 'the', 'of', 'method' or 'system' are thrown away because they carry no topic meaning.
# Takes the useful words out of a concept and drops fillers.
def _concept_words(concept: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", str(concept).lower())
    return [w for w in words if w not in _CONCEPT_STOPWORDS and len(w) > 1]


# Checks if a paper really talks about one concept.
# Steps: take the important words of the concept; if the paper text contains the acronym (for example GNN) say yes;
# otherwise count how many of the words (or their root forms) occur in the paper.
# A concept of up to 3 words needs all words; longer concepts need about 75 percent of them.
# Does the paper really talk about this concept?
def _paper_covers_concept(text: str, concept: str) -> bool:
    """True if the paper text contains the concept's content words (or its acronym, e.g. GNN)."""
    words = _concept_words(concept)
    if not words:
        return True
    tokens = set(re.findall(r"[a-z0-9]+", text))
    stems = {_stem(t) for t in tokens}
    if len(words) >= 2 and "".join(w[0] for w in words) in tokens:      # acronym such as gnn, llm
        return True
    hits = sum(1 for w in words if w in tokens or _stem(w) in stems)
    needed = len(words) if len(words) <= 3 else int(len(words) * 0.75 + 0.999)
    return hits >= needed


# Topic relevance gate: a rule-based check that stops papers from a NEARBY field from being selected.
# The similarity scores alone can rank an off-topic paper high just because it shares broad words like 'AI agents'.
# Steps:
#  1. Build the paper text and the keyword groups from the query (_query_anchor_groups).
#  2. If the query is about a literature-review task, the paper must contain the review idea (and multi-agent if asked).
#  3. For ordinary topics, the paper must cover all core concepts found by Query Understanding
#     (or at least half of them if its semantic score is high).
#  4. If no concepts exist, a very strong title match can still let the paper pass.
# Returns passed (True/False), a short reason, and which groups matched or were missing.
# The bouncer. It checks whether a paper really fits the topic or is only nearby.
def _topic_relevance_gate(
    paper: Dict[str, Any],
    original_query: str,
    research_intent: str,
    core_concepts: List[Any],
    subtopics: List[Any],
    related_concepts: List[Any],
    semantic_score: float,
    tfidf_score: float,
    title_score: float,
) -> Dict[str, Any]:
    """
    Deterministic topic-fit gate.

    The existing hybrid ranker remains the main ranking mechanism. This gate
    exists specifically to stop an adjacent application-domain paper from
    winning simply because it contains broad terms such as "AI agents".

    For compound topics, the gate requires the paper to preserve the important
    conceptual anchors. For example, a query containing both "multi-agent" and
    "literature review generation" should not select an agriculture or vehicle
    paper that only happens to discuss agentic AI.
    """
    text = _paper_text_for_relevance(paper)
    title = str(paper.get("title") or "").lower()
    groups = _query_anchor_groups(
        original_query,
        research_intent,
        core_concepts,
        subtopics,
        related_concepts,
    )

    matched_groups = []
    missing_groups = []
    group_scores = {}

    for group_name, terms in groups.items():
        title_match = _contains_any_term(title, terms)
        text_match = _contains_any_term(text, terms)
        score = 1.0 if title_match else (0.75 if text_match else 0.0)
        group_scores[group_name] = score
        if text_match:
            matched_groups.append(group_name)
        else:
            missing_groups.append(group_name)

    # Strong semantic/title evidence can rescue a candidate for a simple query,
    # but compound research tasks require their defining task anchor.
    compound_anchor_required = "review_task" in groups
    has_review_anchor = "review_task" in matched_groups
    has_multi_agent_anchor = "multi_agent" not in groups or "multi_agent" in matched_groups

    # When the user explicitly asks for literature-review generation, a paper
    # about generic agentic AI is not sufficiently on-topic without a review,
    # synthesis, or scholarly-review signal.
    if compound_anchor_required:
        passed = has_review_anchor and has_multi_agent_anchor
        if not passed:
            reason = "missing required compound-topic anchor(s)"
        elif title_score >= 0.30 or tfidf_score >= 0.10 or semantic_score >= 0.35:
            passed = True
            reason = "required compound-topic anchors matched"
        else:
            passed = False
            reason = "anchors matched but overall evidence is weak"
    else:
        # Generic topics: the paper must cover the core concepts found by Query Understanding
        # (e.g. both "graph neural networks" and "traffic forecasting"), so that a paper that
        # shares only one concept or a broad field is not accepted as on-topic.
        concepts = [c for c in _unique_strings(core_concepts) if _concept_words(c)]
        covered = [c for c in concepts if _paper_covers_concept(text, c)]
        coverage = len(covered) / len(concepts) if concepts else None
        matched_count = len(matched_groups)
        if coverage is not None and coverage < 1.0 and not (coverage >= 0.5 and semantic_score >= 0.55):
            passed = False
            reason = f"covers {len(covered)} of {len(concepts)} core concepts"
        elif groups and matched_count == 0 and semantic_score < 0.35 and tfidf_score < 0.08:
            passed = False
            reason = "no deterministic topic anchors and weak similarity"
        else:
            passed = True
            reason = (f"covers all {len(concepts)} core concepts" if coverage == 1.0
                      else "topic fit supported by hybrid retrieval signals")

    # A paper can also be a very strong direct title match even if Query
    # Understanding failed to produce a useful concept list.
    if not groups and title_score >= 0.55:
        passed = True
        reason = "strong direct title match"

    return {
        "passed": bool(passed),
        "reason": reason,
        "matched_groups": matched_groups,
        "missing_groups": missing_groups,
        "group_scores": group_scores,
    }


# AGENT 1 - SEARCH: chooses which papers will be reviewed. It reads topic, years and max_papers from the state.
# Steps:
#  1. Query understanding: an LLM turns the topic into intent, core concepts and up to 10 search queries
#     (the exact user topic is always kept).
#  2. retrieve_and_rank(): asks the Node server for candidates from MongoDB, removes preprints and papers without a PDF,
#     scores every candidate and applies the relevance gate.
#  3. If MongoDB gave fewer papers than requested, search OpenAlex online, save new papers to MongoDB and rank again.
#  4. Keep the best max_papers and store them in state.raw_papers_list.
# If nothing relevant is found, it records an error and returns an empty list.
# Finds the best N papers. It makes search phrases, asks Node for papers from MongoDB, scores them, 
# filters them with the bouncer, and goes to OpenAlex if there are too few.
def search_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    Domain-agnostic scholarly retrieval pipeline.

    Pipeline:
        User topic
            ↓
        Query Understanding Agent
            ↓
        Multiple targeted scholarly queries
            ↓
        MongoDB multi-query candidate retrieval
            ↓
        Semantic + TF-IDF + title hybrid ranking
            ↓
        Deterministic compound-topic relevance gate
            ↓
        Top-N papers
    """

    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    print(f"\n🔎 [Search Agent]: Searching for '{state.topic}'")
    print(f"📚 [Search Agent]: Target papers = {state.max_papers}")

    try:
        # =========================================================
        # 1. VALIDATE QUERY
        # =========================================================
        original_query = state.topic.strip()

        if not original_query:
            return {
                "raw_papers_list": [],
                "error": "Research topic cannot be empty."
            }

        # =========================================================
        # 2. QUERY UNDERSTANDING + EXPANSION
        # =========================================================
        print(
            "\n🧠 [Search Agent]: "
            "Running Query Understanding Agent..."
        )

        try:
            intent = understand_research_query(original_query)

            search_queries = intent.get("search_queries", [])
            research_intent = intent.get("research_intent", original_query)
            core_concepts = intent.get("core_concepts", [])
            domain = intent.get("domain", [])
            subtopics = intent.get("subtopics", [])
            related_concepts = intent.get("related_concepts", [])

        except Exception as intent_error:
            print(
                f"⚠️ [Search Agent]: Query understanding failed: {intent_error}"
            )
            print("↪ Falling back to original research query.")

            search_queries = [original_query]
            research_intent = original_query
            core_concepts = []
            domain = []
            subtopics = []
            related_concepts = []

        # =========================================================
        # 3. CLEAN AND STRENGTHEN SEARCH QUERIES
        # =========================================================
        cleaned_queries = _unique_strings(search_queries)

        # Always retain the exact user query.
        cleaned_queries = [
            original_query,
            *[q for q in cleaned_queries if q.lower() != original_query.lower()],
        ]

        # Deterministic compound-query safeguards. These are added only when
        # the user's wording clearly contains the corresponding concept.
        lower_original = original_query.lower()

        if any(term in lower_original for term in [
            "literature review", "literature-review", "review generation",
            "automated literature", "automatic literature", "systematic review",
        ]):
            strengthened = [
                f"{original_query} academic papers",
                f"{original_query} scholarly literature",
                f"{original_query} research synthesis",
                "multi-agent literature review generation",
                "multi-agent automated literature review",
                "LLM literature review generation",
                "automated scholarly literature synthesis",
            ]
            cleaned_queries = _unique_strings(strengthened + cleaned_queries)

        if any(term in lower_original for term in [
            "multi-agent", "multi agent", "multiagent", "multiple agents",
        ]):
            strengthened = [
                f"{original_query} multi-agent systems",
                f"{original_query} agent collaboration",
            ]
            cleaned_queries = _unique_strings(strengthened + cleaned_queries)

        search_queries = cleaned_queries[:10]

        print(
            "\n🧩 [Search Agent]: Targeted scholarly search queries"
        )
        print("=" * 100)
        print(f"Research intent : {research_intent}")
        print(
            f"Domain          : "
            f"{', '.join(map(str, domain)) if domain else 'Not explicitly identified'}"
        )
        print(
            f"Core concepts   : "
            f"{', '.join(map(str, core_concepts)) if core_concepts else 'None'}"
        )
        print("-")
        for i, search_query in enumerate(search_queries, start=1):
            print(f"{i}. {search_query}")
        print("=" * 100)

        # Inner helper of the Search agent: from queries to a ranked, filtered list of papers.
        #  1. Calls the Node endpoint /api/corpus/search with the queries and year range (asks for at least 100 candidates).
        #  2. Drops papers without a PDF link and papers hosted on preprint sites (arXiv, SSRN, ...); removes duplicates.
        #  3. Scores each candidate: 60% meaning similarity (MiniLM embeddings), 20% TF-IDF word overlap,
        #     15% title match, 5% metadata completeness.
        #  4. Runs the topic relevance gate; a few strong near-misses may be 'rescued' if too few papers pass.
        #  5. Sorts by score and returns (candidates, passed, rejected, selected top-N).
        def retrieve_and_rank():
            """Steps 4-12: corpus retrieval from MongoDB, hybrid ranking and topic gate."""
            # =========================================================
            # 4. MONGODB MULTI-QUERY SCHOLARLY CORPUS RETRIEVAL
            # =========================================================
            logger.info(
                f"Retrieving published scholarly literature from MongoDB "
                f"for {state.topic} ({state.start_year}-{state.end_year})"
            )

            # Candidate pool must be considerably larger than final N so that the
            # relevance gate can remove adjacent papers without starving retrieval.
            corpus_limit = max(100, state.max_papers * 25)

            params_dict = {
                "topic": original_query,
                "queries": json.dumps(search_queries, ensure_ascii=False),
                "research_intent": str(research_intent),
                "core_concepts": json.dumps(core_concepts, ensure_ascii=False),
                "domain": json.dumps(domain, ensure_ascii=False),
                "subtopics": json.dumps(subtopics, ensure_ascii=False),
                "related_concepts": json.dumps(related_concepts, ensure_ascii=False),
                "start_year": state.start_year,
                "end_year": state.end_year,
                "limit": corpus_limit,
            }

            params = urllib.parse.urlencode(params_dict)
            corpus_url = f"http://127.0.0.1:3000/api/corpus/search?{params}"

            try:
                with urllib.request.urlopen(corpus_url, timeout=60) as response:
                    payload = json.loads(response.read().decode("utf-8"))

                corpus_papers = payload.get("papers", [])

                logger.info(
                    f"MongoDB corpus returned {len(corpus_papers)} candidates"
                )
                print(
                    f"\n🔎 [MongoDB Retrieval] Returned "
                    f"{len(corpus_papers)} candidates"
                )

                if payload.get("queries"):
                    print(
                        f"   Multi-query retrieval used "
                        f"{len(payload.get('queries', []))} query variants."
                    )

                if corpus_papers:
                    print(
                        f"   First candidate: "
                        f"{corpus_papers[0].get('title', '')[:100]}"
                    )
                else:
                    print(f"   ⚠️ MongoDB response payload: {payload}")

                # Explicitly reject known preprint repositories.
                blocked_hosts = (
                    "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com",
                    "engrxiv.org", "chemrxiv.org", "researchsquare.com",
                )

                normalized = []
                seen_ids = set()

                for paper in corpus_papers:
                    pdf_url = str(paper.get("pdf_url") or "").strip()
                    paper_url = str(paper.get("url") or "").strip()
                    combined_urls = (pdf_url + " " + paper_url).lower()

                    if not pdf_url:
                        continue

                    if any(host in combined_urls for host in blocked_hosts):
                        continue

                    paper_id = (
                        paper.get("paper_id")
                        or paper.get("doi")
                        or paper.get("_id")
                        or paper_url
                    )
                    paper_id = str(paper_id)

                    if paper_id in seen_ids:
                        continue
                    seen_ids.add(paper_id)

                    normalized.append({
                        "paper_id": paper_id,
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "year": paper.get("year"),
                        "abstract": paper.get("abstract", ""),
                        "doi": paper.get("doi"),
                        "url": paper_url,
                        "pdf_url": pdf_url,
                        "source": paper.get("source", "MongoDB scholarly corpus"),
                        "retrieval_queries": paper.get("retrieval_queries") or search_queries,
                        "research_intent": research_intent,
                        "domain": paper.get("domain") or domain,
                        "core_concepts": paper.get("core_concepts") or core_concepts,
                        "subtopics": paper.get("subtopics") or subtopics,
                        "related_concepts": paper.get("related_concepts") or related_concepts,
                        "venue": paper.get("venue", ""),
                    })

                state.raw_papers_list = normalized
                candidates = normalized

                logger.info(
                    f"Eligible non-preprint scholarly papers: {len(candidates)}"
                )
                print(
                    f"   Eligible non-preprint PDF papers: {len(candidates)}"
                )

                if not candidates:
                    return [], [], [], []

            except Exception as e:
                logger.exception("MongoDB scholarly corpus retrieval failed")
                state.raw_papers_list = []
                print(f"⚠️ [Search Agent]: MongoDB corpus retrieval failed: {e}")
                return [], [], [], []

            # =========================================================
            # 5. DOCUMENT TEXT
            # =========================================================
            document_texts = []
            for paper in candidates:
                title = paper.get("title", "")
                abstract = paper.get("abstract", "")
                document_texts.append(f"{title}. {abstract}")

            # =========================================================
            # 6. SEMANTIC RANKING
            # =========================================================
            semantic_scores = np.zeros(len(candidates), dtype=float)
            semantic_available = False

            try:
                print("\n🧠 [Search Agent]: Loading semantic model...")
                model = SentenceTransformer("all-MiniLM-L6-v2")

                query_embeddings = model.encode(
                    search_queries,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                document_embeddings = model.encode(
                    document_texts,
                    normalize_embeddings=True,
                    batch_size=16,
                    show_progress_bar=False,
                )

                similarity_matrix = document_embeddings @ query_embeddings.T
                semantic_scores = similarity_matrix.max(axis=1)
                semantic_available = True

                print("✅ [Search Agent]: Multi-query semantic ranking available.")

            except Exception as embedding_error:
                print("⚠️ [Search Agent]: Semantic model unavailable.")
                print(f"   Reason: {embedding_error}")
                print("↪ Falling back to TF-IDF ranking.")

            # =========================================================
            # 7. TF-IDF RANKING
            # =========================================================
            tfidf_scores = np.zeros(len(candidates), dtype=float)

            try:
                combined_query = " ".join(search_queries)
                tfidf_vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(
                    [combined_query] + document_texts
                )
                tfidf_scores = cosine_similarity(
                    tfidf_matrix[0:1], tfidf_matrix[1:]
                )[0]

            except Exception as tfidf_error:
                print(f"⚠️ [Search Agent]: TF-IDF scoring failed: {tfidf_error}")

            # =========================================================
            # 8. TITLE RELEVANCE
            # =========================================================
            title_scores = np.zeros(len(candidates), dtype=float)

            try:
                title_vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                )
                combined_query = " ".join(search_queries)
                title_matrix = title_vectorizer.fit_transform(
                    [combined_query] + [paper["title"] for paper in candidates]
                )
                title_scores = cosine_similarity(
                    title_matrix[0:1], title_matrix[1:]
                )[0]

            except Exception:
                title_scores = np.zeros(len(candidates), dtype=float)

            # =========================================================
            # 9. METADATA QUALITY
            # =========================================================
            quality_scores = np.zeros(len(candidates), dtype=float)

            for i, paper in enumerate(candidates):
                score = 0.0
                if paper.get("title"):
                    score += 0.20
                if paper.get("abstract"):
                    score += 0.35
                if paper.get("authors"):
                    score += 0.15
                if paper.get("year"):
                    score += 0.15
                if paper.get("doi") or paper.get("arxiv_id"):
                    score += 0.15
                quality_scores[i] = score

            # =========================================================
            # 10. NORMALIZATION
            # =========================================================
            # Rescales a list of scores to the range 0..1 (lowest becomes 0, highest becomes 1)
            # so that different score types can be mixed fairly. If all scores are equal, every score becomes 0.5.
            def normalize_scores(scores):
                scores = np.asarray(scores, dtype=float)
                if len(scores) == 0:
                    return scores
                min_value = scores.min()
                max_value = scores.max()
                if max_value - min_value < 1e-9:
                    return np.ones_like(scores) * 0.5
                return (scores - min_value) / (max_value - min_value)

            semantic_norm = normalize_scores(semantic_scores)
            tfidf_norm = normalize_scores(tfidf_scores)
            title_norm = normalize_scores(title_scores)
            quality_norm = normalize_scores(quality_scores)

            # =========================================================
            # 11. HYBRID RANKING + TOPIC RELEVANCE GATE
            # =========================================================
            scored_papers = []
            rejected_by_gate = []

            for i, paper in enumerate(candidates):
                if semantic_available:
                    final_score = (
                        0.60 * semantic_norm[i]
                        + 0.20 * tfidf_norm[i]
                        + 0.15 * title_norm[i]
                        + 0.05 * quality_norm[i]
                    )
                else:
                    final_score = (
                        0.70 * tfidf_norm[i]
                        + 0.20 * title_norm[i]
                        + 0.10 * quality_norm[i]
                    )

                paper["semantic_score"] = round(float(semantic_scores[i]), 4)
                paper["tfidf_score"] = round(float(tfidf_scores[i]), 4)
                paper["title_score"] = round(float(title_scores[i]), 4)
                paper["metadata_quality_score"] = round(float(quality_scores[i]), 4)
                paper["relevance_score"] = round(float(final_score), 4)

                gate = _topic_relevance_gate(
                    paper=paper,
                    original_query=original_query,
                    research_intent=str(research_intent),
                    core_concepts=core_concepts,
                    subtopics=subtopics,
                    related_concepts=related_concepts,
                    semantic_score=float(semantic_scores[i]),
                    tfidf_score=float(tfidf_scores[i]),
                    title_score=float(title_scores[i]),
                )

                paper["topic_relevance_pass"] = gate["passed"]
                paper["topic_relevance_reason"] = gate["reason"]
                paper["topic_relevance_matched_groups"] = gate["matched_groups"]
                paper["topic_relevance_missing_groups"] = gate["missing_groups"]

                reasons = []
                if semantic_available:
                    if semantic_norm[i] >= 0.70:
                        reasons.append("strong semantic similarity")
                    elif semantic_norm[i] >= 0.45:
                        reasons.append("moderate semantic similarity")
                if tfidf_norm[i] >= 0.60:
                    reasons.append("strong terminology overlap")
                if title_norm[i] >= 0.50:
                    reasons.append("focused title match")
                if quality_scores[i] >= 0.80:
                    reasons.append("complete metadata")
                if gate["passed"]:
                    reasons.append("compound-topic relevance gate passed")
                else:
                    reasons.append("compound-topic relevance gate rejected")

                paper["relevance_reason"] = "; ".join(reasons)

                if gate["passed"]:
                    scored_papers.append(paper)
                else:
                    rejected_by_gate.append(paper)

            # If the strict gate leaves too few papers, use a controlled rescue
            # pass only for candidates with genuinely strong direct evidence.
            if len(scored_papers) < state.max_papers:
                rescue_candidates = [
                    p for p in rejected_by_gate
                    if (
                        # a paper missing a core concept of the topic is never rescued
                        not str(p.get("topic_relevance_reason", "")).startswith("covers ")
                        and p.get("title_score", 0.0) >= 0.55
                        and p.get("semantic_score", 0.0) >= 0.45
                    )
                ]
                rescue_candidates.sort(
                    key=lambda paper: paper.get("relevance_score", 0.0),
                    reverse=True,
                )
                for paper in rescue_candidates:
                    paper["topic_relevance_pass"] = True
                    paper["topic_relevance_reason"] = (
                        "rescued by exceptionally strong direct semantic/title evidence"
                    )
                    paper["relevance_reason"] += "; controlled relevance rescue"
                    scored_papers.append(paper)
                    if len(scored_papers) >= state.max_papers:
                        break

            scored_papers.sort(
                key=lambda paper: paper["relevance_score"],
                reverse=True,
            )

            # =========================================================
            # 12. SELECT TOP N
            # =========================================================
            selected_papers = scored_papers[:state.max_papers]

            # Final safety: never return a paper that failed the topic gate.
            selected_papers = [
                paper for paper in selected_papers
                if paper.get("topic_relevance_pass") is True
            ][:state.max_papers]


            return candidates, scored_papers, rejected_by_gate, selected_papers

        candidates, scored_papers, rejected_by_gate, selected_papers = retrieve_and_rank()

        # =========================================================
        # 12b. ONLINE FALLBACK: database first, then the internet
        # =========================================================
        # If MongoDB does not hold enough relevant papers, fetch candidates from OpenAlex,
        # store them in MongoDB and rank again, so they compete with the stored papers on
        # equal terms and remain available for future reviews.
        if len(selected_papers) < state.max_papers and ONLINE_SEARCH_ENABLED:
            print(
                f"\n🌐 [Search Agent]: MongoDB gave {len(selected_papers)} of "
                f"{state.max_papers} relevant papers; searching OpenAlex online..."
            )
            online = fetch_and_store(search_queries, state.start_year, state.end_year)
            for message in online["errors"]:
                print(f"   ⚠️ {message}")
            print(
                f"   OpenAlex returned {online['found']} usable papers: "
                f"{online['inserted']} new, {online['updated']} already in MongoDB."
            )
            if online["inserted"]:
                candidates, scored_papers, rejected_by_gate, selected_papers = retrieve_and_rank()

        if not selected_papers:
            state.errors.append(
                "No relevant peer-reviewed papers with PDFs were found in MongoDB or OpenAlex."
            )
            return {"raw_papers_list": []}

        # =========================================================
        # 13. DISPLAY
        # =========================================================
        print("\n📊 [Search Agent]: Multi-query hybrid ranking + topic gate")
        print("=" * 120)

        print(
            f"Candidates retrieved : {len(candidates)} | "
            f"Gate passed : {len(scored_papers)} | "
            f"Gate rejected : {len(rejected_by_gate)}"
        )

        for rank, paper in enumerate(scored_papers, start=1):
            selected = rank <= len(selected_papers)
            status = "✓ SELECTED" if selected else "✗ FILTERED"
            print(
                f"{rank:02d}. {paper['relevance_score']:.4f} "
                f"{status} | {paper['title']}"
            )
            if selected:
                print(
                    f"    Semantic : {paper['semantic_score']:.4f} "
                    f"| TF-IDF : {paper['tfidf_score']:.4f} "
                    f"| Title : {paper['title_score']:.4f} "
                    f"| Quality : {paper['metadata_quality_score']:.4f}"
                )
                print(
                    f"    Topic gate: {paper['topic_relevance_reason']}"
                )

        if rejected_by_gate:
            print("\n🚫 [Topic Relevance Gate]: Rejected adjacent-domain candidates")
            for paper in sorted(
                rejected_by_gate,
                key=lambda p: p.get("relevance_score", 0.0),
                reverse=True,
            )[:10]:
                print(
                    f"   - {paper.get('title', '')} "
                    f"| {paper.get('topic_relevance_reason', '')}"
                )

        print("=" * 120)
        print(
            f"✅ [Search Agent]: Selected {len(selected_papers)} "
            f"topic-relevant papers."
        )

        # Preserve only selected papers as the actual downstream corpus.
        state.raw_papers_list = selected_papers

        return {
            "raw_papers_list": selected_papers
        }

    except Exception as e:
        print(f"❌ [Search Agent]: Unexpected retrieval error: {e}")
        logger.exception("Unexpected retrieval error")
        return {
            "raw_papers_list": [],
            "error": str(e),
        }

# Used only by ablation M1 (retrieval + writer, no analysis).
# Wraps each paper's abstract into the analysis format without calling any LLM, so the writer
# can run with raw abstracts only and we can measure how much the evidence analysis helps.
def abstract_only_analyses(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ablation M1: wrap raw abstracts in the analysis schema without any LLM call."""
    analyzed = []
    for paper in papers:
        abstract = str(paper.get("abstract") or paper.get("full_text") or "").strip()
        analyzed.append({
            **paper,
            "paper_analysis": {
                "study_type": "unknown",
                "research_problem": abstract[:2000],
                "objectives": [],
                "methodology": [],
                "dataset": [],
                "models_and_techniques": [],
                "evaluation_metrics": [],
                "key_findings": [],
                "contributions": [],
                "limitations": [],
                "future_work": [],
                "evidence": [],
                "evidence_records": [],
            },
        })
    return analyzed


# =====================================================================
# SECTION 2 — FULL-TEXT EXTRACTION
# =====================================================================

# AGENT 2 - PDF EXTRACTION: gets the full text of every selected paper.
# It calls extract_papers_content(), which downloads the PDFs into paper_cache/ and reads their text.
# In ablation M1 it is skipped and only the abstract is used. Result goes to state.extracted_papers.
# Downloads each PDF and reads its text. If that fails, it uses the abstract.
def paper_extraction_agent_node(
    state: ResearchState
) -> Dict[str, Any]:

    print(
        f"\n📄 [Paper Extraction Agent]: "
        f"Extracting full text from "
        f"{len(state.raw_papers_list)} selected papers..."
    )

    if not state.raw_papers_list:
        print(
            "⚠️ [Paper Extraction Agent]: "
            "No papers available."
        )

        return {
            "extracted_papers": []
        }

    if state.ablation_mode == "M1_retrieval_only":
        # Ablation M1: no full-text evidence pipeline, abstracts only.
        extracted = [
            {
                **paper,
                "content_source": "abstract",
                "full_text": str(paper.get("abstract", "") or ""),
            }
            for paper in state.raw_papers_list
        ]
        print("   ⏭️ [Ablation M1]: Skipping PDF extraction (abstract only).")
        return {"extracted_papers": extracted}

    extracted = extract_papers_content(
        state.raw_papers_list
    )

    print(
        f"✅ [Paper Extraction Agent]: "
        f"Processed {len(extracted)} papers."
    )

    return {
        "extracted_papers": extracted
    }


# =====================================================================
# SECTION 2 — PAPER ANALYSIS
# =====================================================================

# AGENT 3 - PAPER ANALYSIS: turns each paper's text into structured evidence.
# It calls analyze_papers(), where an LLM reads chunks of each paper and extracts methodology, dataset,
# results, metrics and limitations. Result goes to state.analyzed_papers (the Evidence Matrix uses it).
# In ablation M1 it is skipped.
# 	The AI reads the paper text and extracts method, results, metrics and limitations.
def paper_analysis_agent_node(
    state: ResearchState
) -> Dict[str, Any]:

    print(
        f"\n🧠 [Paper Analysis Agent]: "
        f"Analyzing {len(state.extracted_papers)} papers..."
    )

    if not state.extracted_papers:
        print(
            "⚠️ [Paper Analysis Agent]: "
            "No extracted papers available."
        )

        return {
            "analyzed_papers": []
        }

    if state.ablation_mode == "M1_retrieval_only":
        print("   ⏭️ [Ablation M1]: Skipping evidence analysis (raw abstracts passed to writer).")
        return {"analyzed_papers": abstract_only_analyses(state.extracted_papers)}

    analyzed = analyze_papers(
        state.extracted_papers
    )

    print(
        f"✅ [Paper Analysis Agent]: "
        f"Analyzed {len(analyzed)} papers."
    )

    return {
        "analyzed_papers": analyzed
    }


# =====================================================================
# SECTION 3 — CROSS-PAPER COMPARATIVE ANALYSIS
# =====================================================================

# AGENT 4 - CROSS-PAPER ANALYSIS: compares the papers with each other.
# It calls analyze_cross_papers(), which finds common themes, agreements, disagreements and trade-offs.
# Result goes to state.cross_paper_analysis. Skipped in ablation M1 and M2.
# The AI compares the papers: agreements, disagreements, themes.
def cross_paper_analysis_agent_node(
    state: ResearchState
) -> Dict[str, Any]:

    print(
        f"\n🔗 [Cross-Paper Analysis Agent]: "
        f"Comparing {len(state.analyzed_papers)} papers..."
    )

    if not state.analyzed_papers:

        print(
            "⚠️ [Cross-Paper Analysis Agent]: "
            "No analyzed papers available."
        )

        return {
            "cross_paper_analysis": {}
        }

    if state.ablation_mode in {"M1_retrieval_only", "M2_no_cross_paper"}:
        print(f"   ⏭️ [Ablation {state.ablation_mode[:2]}]: Cross-paper synthesis disabled.")
        return {"cross_paper_analysis": {}}

    comparison = analyze_cross_papers(
        state.analyzed_papers
    )

    print(
        "✅ [Cross-Paper Analysis Agent]: "
        "Cross-paper synthesis completed."
    )

    return {
        "cross_paper_analysis": comparison
    }


# =====================================================================
# SECTION 3 — EVIDENCE-GROUNDED RESEARCH-GAP DISCOVERY & VERIFICATION
# =====================================================================

_DETAIL_SCALES = (1.0, 0.6, 0.4, 0.25, 0.15)
_CROSS_CHARS = (12000, 6000, 3500, 2000, 1000)


# Builds a SHORT evidence pack for the research-gap agents.
# For every analysed paper it copies title, year, problem, methodology, techniques, findings, limitations,
# future work and evidence, then cuts each list to a few items. 'level' makes the pack smaller step by step
# (level 0 = most detail). The cross-paper analysis is added too, trimmed at higher levels.
# A smaller pack is needed because Groq's free tier allows only a limited number of tokens per minute.
# Makes a short evidence pack for the gap debate.
def _compact_gap_context(state: ResearchState, level: int = 0) -> Dict[str, Any]:
    """Build compact, source-grounded evidence for the gap agents (level 0 = most detail)."""

    scale = _DETAIL_SCALES[min(level, len(_DETAIL_SCALES) - 1)]

    # Tiny helper: shrinks a list length according to the current detail level (never below 1).
    def lim(n: int) -> int:
        return max(1, int(round(n * scale)))

    papers = []

    for paper in state.analyzed_papers:
        analysis = paper.get("paper_analysis", {}) or {}

        papers.append({
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "research_problem": str(analysis.get("research_problem", ""))[:400],
            "methodology": analysis.get("methodology", [])[:lim(6)],
            "models_and_techniques": analysis.get("models_and_techniques", [])[:lim(8)],
            "key_findings": analysis.get("key_findings", [])[:lim(8)],
            "limitations": analysis.get("limitations", [])[:lim(8)],
            "future_work": analysis.get("future_work", [])[:lim(8)],
            "evidence": analysis.get("evidence", [])[:lim(12)],
        })

    cross_analysis = state.cross_paper_analysis

    if isinstance(cross_analysis, dict):
        compact_cross = {
            key: value
            for key, value in cross_analysis.items()
            if key not in {"raw_text", "full_text", "paper_details"}
        }
        if level > 0:
            compact_cross = json.dumps(compact_cross, ensure_ascii=False)[: _CROSS_CHARS[min(level, 4)]]
    else:
        compact_cross = str(cross_analysis or "")[: _CROSS_CHARS[min(level, 4)]]

    return {
        "topic": state.topic,
        "papers": papers,
        "cross_paper_analysis": compact_cross,
    }


# Makes a gap-agent prompt small enough for the token limit.
# It estimates the tokens of the prompt; while it is still too big, it swaps the embedded evidence
# for a smaller version (next detail level) until it fits PROMPT_TOKEN_BUDGET. Prints the final size.
def _fit_gap_prompt(prompt: str, state: ResearchState, label: str) -> str:
    """Shrink the embedded evidence until the prompt fits the per-minute token budget."""
    current = json.dumps(_compact_gap_context(state), ensure_ascii=False, indent=2)
    for level in range(1, len(_DETAIL_SCALES)):
        if estimate_tokens(prompt) <= PROMPT_TOKEN_BUDGET:
            break
        smaller = json.dumps(_compact_gap_context(state, level), ensure_ascii=False)
        prompt = prompt.replace(current, smaller)
        current = smaller
    print(f"   📏 [{label}] prompt ≈ {estimate_tokens(prompt)} tokens")
    return prompt


# Creates the language model connection used by the gap agents: gpt-oss-20b on Groq, temperature 0
# (temperature 0 = the most stable, least random answers).
def _gap_llm():
    return make_llm("openai/gpt-oss-20b", temperature=0)


# Finds the JSON object inside an LLM's text answer.
# LLMs often wrap JSON in markdown fences or add chatty sentences, so this function tries several ways:
# remove <think> tags, take the content of a ``` fence, parse the whole text, then scan for the first balanced
# { ... } block, and finally a rough regex. If nothing works it raises an error.
def _extract_json_object(text: str) -> Any:
    """Extract a JSON object from plain LLM output, including fenced or conversational text."""
    if not text:
        raise ValueError("Empty LLM response.")

    cleaned = str(text).strip()

    # Strip <think>...</think> reasoning tags
    cleaned = re.sub(r"(?is)<think>.*?</think>", "", cleaned).strip()

    # If fenced markdown exists, take contents of first fence
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, flags=re.IGNORECASE)
    if fence_match:
        fenced = fence_match.group(1).strip()
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            cleaned = fenced

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Recover the first complete JSON object by scanning bracket balance
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False

        for index in range(start, len(cleaned)):
            char = cleaned[index]

            if in_string:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        sanitized = re.sub(r'[\x00-\x1f]+', ' ', candidate)
                        return json.loads(sanitized)

    # Grep-like fallback: search for outer JSON block
    outer_match = re.search(r"(\{[\s\S]*\})", cleaned)
    if outer_match:
        try:
            return json.loads(outer_match.group(1))
        except Exception:
            pass

    raise ValueError("No valid JSON object found in LLM response.")


# Asks the LLM a gap-stage question and returns a validated, typed answer.
#  1. Adds strict instructions: return ONLY one JSON object that matches the given schema.
#  2. Calls the LLM, extracts the JSON, validates it with the Pydantic schema.
#  3. If it fails, it retries once with a stronger reminder.
#  4. If it still fails it returns None, and the caller switches to a safe evidence-based fallback.
# Quota/rate-limit errors are not swallowed; they stop the whole run.
# Asks the AI, checks the reply shape, and retries once. If it fails, returns nothing so a fallback is used.
def _invoke_gap_json(schema, prompt: str, label: str):
    """
    Invoke a gap-stage LLM without provider tool-calling.
    Returns validated schema instance, or None on failure (enabling deterministic fallback).
    """
    strict_prompt = prompt + """

OUTPUT FORMAT:
Return ONLY one valid JSON object.
Do not use Markdown.
Do not use ``` fences.
Do not add explanations before or after the JSON.
The JSON must conform exactly to this schema:
""" + json.dumps(schema.model_json_schema(), ensure_ascii=False)

    last_error = None

    for attempt in range(2):
        try:
            raw = invoke_text(_gap_llm(), strict_prompt, label=label)
            payload = _extract_json_object(raw)
            validated = schema.model_validate(payload)

            print(
                f"   ✅ [{label}]: JSON response validated "
                f"(attempt {attempt + 1})."
            )
            return validated

        except LLMUnavailable:
            raise
        except Exception as error:
            last_error = error
            print(
                f"   ⚠️ [{label}]: JSON invocation attempt "
                f"{attempt + 1} failed: {error}"
            )

            if attempt == 0:
                strict_prompt += """

RETRY REQUIREMENT:
Your previous response could not be parsed as JSON.
Return the complete JSON object again, with every required field present.
Keep the response compact and syntactically valid.
"""

    print(f"   ⚠️ [{label}]: LLM JSON failed after retries: {last_error}. Switching to evidence-grounded fallback.")
    return None


# Computes how many LLM calls and tokens one stage used: the counters after minus the counters before.
def _usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {key: after.get(key, 0) - before.get(key, 0) for key in after}


# Ablation M3 only: lets every proposed gap through WITHOUT the Skeptic/Verifier.
# It converts each grounded candidate straight into a 'verified' record, so we can measure
# how much the Skeptic and Verifier improve gap quality.
# Used only in mode M3 (no debate): accepts every gap.
def accept_candidates_without_verification(
    grounded_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Ablation M3: pass proposer gaps through without Skeptic/Verifier review."""
    accepted = []
    for candidate in grounded_candidates:
        supporting = list(dict.fromkeys(candidate.get("supporting_papers", [])))
        accepted.append({
            "gap_statement": str(candidate.get("gap_statement", "")).strip(),
            "supporting_papers": supporting,
            "evidence_basis": list(candidate.get("supporting_evidence", []))[:8],
            "verification_reason": "Not verified (Skeptic/Verifier disabled for ablation).",
            "verified": True,
            "support_count": len(supporting),
            "cross_paper_supported": len(supporting) >= 2,
        })
    return accepted


# AGENT 5 - RESEARCH GAP: finds gaps in the literature and checks them like a small debate.
#  1. PROPOSER (LLM): suggests 4-6 candidate gaps using only the supplied evidence. If the LLM fails, gaps are
#     built from the limitations the papers report.
#  2. GROUNDING FILTER: keeps only candidates that name real paper titles and give evidence.
#  3. SKEPTIC (LLM): challenges every candidate (unsupported? only future work? overstated?).
#  4. VERIFIER (LLM): confirms only gaps that survive the Skeptic's concerns.
#  5. FINAL GATE (code): the gap needs 6+ words, at least one real supporting paper and evidence.
# If everything is rejected, nothing is forced through; the writer then discusses only reported limitations.
# Ablation M1 skips this agent; M3 accepts proposals without steps 3-4.
# A small debate. The Proposer suggests gaps, the Skeptic attacks them, the Verifier keeps only the strong ones.
def research_gap_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    Evidence-Grounded Multi-Agent Research-Gap Discovery.

    Proposer -> Challenger/Skeptic -> Verifier -> deterministic
    grounding gate. Only verified gaps are exposed downstream.
    """

    print("\n🔬 [Research Gap Agent]: Starting gap discovery and verification...")

    if state.ablation_mode == "M1_retrieval_only":
        print("   ⏭️ [Ablation M1]: Research-gap agents disabled.")
        return {
            "research_gap_analysis": {
                "status": "DISABLED_ABLATION",
                "proposed_gaps": [],
                "challenged_gaps": [],
                "verified_gaps": [],
            },
            "verified_research_gaps": [],
        }

    if not state.analyzed_papers:
        return {
            "research_gap_analysis": {
                "status": "NO_EVIDENCE",
                "proposed_gaps": [],
                "challenged_gaps": [],
                "verified_gaps": [],
            },
            "verified_research_gaps": [],
        }

    context = _compact_gap_context(state)
    usage_start = snapshot_token_usage()

    paper_titles = {
        str(paper.get("title", "")).strip()
        for paper in state.analyzed_papers
        if str(paper.get("title", "")).strip()
    }

    # ---------------------------------------------------------------
    # 1. PROPOSER
    # ---------------------------------------------------------------

    proposer_prompt = f"""
You are the Research Gap Proposer.

Research topic:
{state.topic}

SOURCE-GROUNDED PAPER EVIDENCE:
{json.dumps(context, ensure_ascii=False, indent=2)}

Identify 4 to 6 candidate research gaps (favour recall: a separate Skeptic
and Verifier will reject candidates that are not supported).

Rules:
1. Use ONLY the supplied evidence.
2. Ground every candidate in explicit limitations, missing evidence,
   disagreements, methodological differences, or unresolved issues.
3. Prefer gaps supported by multiple papers when possible.
4. Do not claim the entire field lacks something because one paper
   does not mention it.
5. Do not introduce outside knowledge.
6. Do not turn future work into a demonstrated limitation.
7. Do not infer experimental weaknesses that are not reported.
8. Use exact paper titles.
9. Keep each gap specific and academically cautious.
"""

    proposer_prompt = _fit_gap_prompt(proposer_prompt, state, "Gap Proposer")

    proposal_result = _invoke_gap_json(
        ResearchGapProposalSchema,
        proposer_prompt,
        "Gap Proposer",
    )

    proposed_candidates = []
    if proposal_result and getattr(proposal_result, "candidates", None):
        proposed_candidates = [
            candidate.model_dump()
            for candidate in proposal_result.candidates
        ]
        print(f"   💡 [Gap Proposer]: Generated {len(proposed_candidates)} candidates via LLM.")
    else:
        # Resilient evidence-grounded synthesis from analyzed papers
        print("   💡 [Gap Proposer]: Synthesizing candidate gaps directly from extracted paper limitations...")
        for p in state.analyzed_papers:
            p_title = str(p.get("title", "")).strip()
            p_analysis = p.get("paper_analysis", {})
            lims = p_analysis.get("limitations", [])
            prob = p_analysis.get("research_problem", "")
            if isinstance(lims, list):
                for lim in lims[:2]:
                    lim_text = str(lim).strip()
                    if lim_text and len(lim_text) > 10:
                        proposed_candidates.append({
                            "gap_statement": f"In {state.topic}, {lim_text.rstrip('.')} remains an unaddressed limitation across deployed methodologies.",
                            "supporting_papers": [p_title],
                            "supporting_evidence": [f"Reported limitation in {p_title}: {lim_text}"],
                            "gap_type": "methodological",
                            "rationale": f"Explicitly identified weakness in {p_title}"
                        })

    # Deterministic source-title/evidence gate before challenge.
    grounded_candidates = []

    for candidate in proposed_candidates:
        supporting = [
            str(title).strip()
            for title in candidate.get("supporting_papers", [])
            if str(title).strip() in paper_titles
        ]

        evidence = [
            str(item).strip()
            for item in candidate.get("supporting_evidence", [])
            if str(item).strip()
        ]

        statement = str(
            candidate.get("gap_statement", "")
        ).strip()

        if statement and supporting and evidence:
            candidate["supporting_papers"] = list(dict.fromkeys(supporting))
            candidate["supporting_evidence"] = evidence[:8]
            grounded_candidates.append(candidate)

    # NOTE: no generic "template" gaps are fabricated when grounding removes
    # every candidate - an empty result is reported honestly instead.

    print(
        f"   🔎 [Grounding Filter]: "
        f"{len(grounded_candidates)} candidates retained."
    )

    if state.ablation_mode == "M3_no_skeptic":
        accepted = accept_candidates_without_verification(grounded_candidates)
        print(
            f"   ⏭️ [Ablation M3]: Skeptic/Verifier disabled - "
            f"{len(accepted)} proposer gaps accepted unverified."
        )
        return {
            "research_gap_analysis": {
                "status": "UNVERIFIED_ABLATION",
                "proposed_gaps": grounded_candidates,
                "challenged_gaps": [],
                "verified_gaps": accepted,
                "stage_usage": {"proposer": _usage_delta(usage_start, snapshot_token_usage())},
            },
            "verified_research_gaps": accepted,
        }

    usage_after_proposer = snapshot_token_usage()

    # ---------------------------------------------------------------
    # 2. CHALLENGER / SKEPTIC
    # ---------------------------------------------------------------

    challenger_prompt = f"""
You are the skeptical Challenger in a research-gap verification pipeline.

Research topic:
{state.topic}

SOURCE PAPER TITLES:
{json.dumps(sorted(paper_titles), ensure_ascii=False, indent=2)}

SOURCE EVIDENCE:
{json.dumps(context, ensure_ascii=False, indent=2)}

CANDIDATE GAPS:
{json.dumps(grounded_candidates, ensure_ascii=False, indent=2)}

Challenge every candidate.
Check:
1. Is the gap directly supported by the supplied evidence?
2. Does it rely on an unsupported field-wide claim?
3. Is it merely a paper's future-work suggestion?
4. Are the cited papers genuinely relevant?
5. Does the wording overstate certainty?
"""

    challenger_prompt = _fit_gap_prompt(challenger_prompt, state, "Gap Challenger")

    challenge_result = _invoke_gap_json(
        ResearchGapChallengeSchema,
        challenger_prompt,
        "Gap Challenger",
    )

    challenged_gaps = []
    if challenge_result and getattr(challenge_result, "challenges", None):
        challenged_gaps = [
            challenge.model_dump()
            for challenge in challenge_result.challenges
        ]
        print(f"   🧐 [Gap Challenger]: Reviewed {len(challenged_gaps)} candidates via LLM.")
    else:
        # The Skeptic did not return a usable assessment: record that honestly instead of
        # inventing a critique, so the Verifier and the UI know these candidates were not challenged.
        print("   ⚠️ [Gap Challenger]: No skeptic assessment returned; candidates marked as not reviewed.")
        for cand in grounded_candidates:
            challenged_gaps.append({
                "gap_statement": cand.get("gap_statement", ""),
                "challenge_status": "NOT_REVIEWED",
                "challenge_valid": None,
                "concerns": [],
                "unsupported_claims": [],
                "required_evidence": [],
                "revised_gap_statement": "",
            })

    # ---------------------------------------------------------------
    # 3. VERIFIER
    # ---------------------------------------------------------------

    verifier_prompt = f"""
You are the final Research Gap Verifier.

Research topic:
{state.topic}

SOURCE PAPER TITLES:
{json.dumps(sorted(paper_titles), ensure_ascii=False, indent=2)}

SOURCE EVIDENCE:
{json.dumps(context, ensure_ascii=False, indent=2)}

PROPOSED CANDIDATES:
{json.dumps(grounded_candidates, ensure_ascii=False, indent=2)}

CHALLENGER ASSESSMENTS:
{json.dumps(challenged_gaps, ensure_ascii=False, indent=2)}

Verify only genuinely supported gaps.
"""

    verifier_prompt = _fit_gap_prompt(verifier_prompt, state, "Gap Verifier")

    verification_result = _invoke_gap_json(
        ResearchGapVerificationSchema,
        verifier_prompt,
        "Gap Verifier",
    )

    verified_candidates = []
    if verification_result and getattr(verification_result, "verified_gaps", None):
        verified_candidates = [
            gap.model_dump()
            for gap in verification_result.verified_gaps
        ]
        print(f"   ✅ [Gap Verifier]: Received {len(verified_candidates)} verification results.")
    else:
        # No Verifier output: accept only candidates the Skeptic explicitly let through, and say so.
        survived = {
            str(ch.get("gap_statement", "")).strip()
            for ch in challenged_gaps
            if ch.get("challenge_valid") is True
        }
        print(f"   ⚠️ [Gap Verifier]: No verifier output; keeping {len(survived)} Skeptic-approved candidates.")
        for cand in grounded_candidates:
            if str(cand.get("gap_statement", "")).strip() not in survived:
                continue
            verified_candidates.append({
                "gap_statement": cand.get("gap_statement", ""),
                "supporting_papers": cand.get("supporting_papers", []),
                "evidence_basis": cand.get("supporting_evidence", []),
                "verification_reason": "Accepted by the Skeptic; the Verifier returned no assessment.",
                "verified": True
            })

    # ---------------------------------------------------------------
    # 4. FINAL DETERMINISTIC VERIFICATION GATE
    # ---------------------------------------------------------------

    final_verified = []

    for gap in verified_candidates:
        if gap.get("verified") is not True:
            continue

        statement = str(
            gap.get("gap_statement", "")
        ).strip()

        supporting_papers = [
            str(title).strip()
            for title in gap.get("supporting_papers", [])
            if str(title).strip() in paper_titles
        ]

        evidence_basis = [
            str(item).strip()
            for item in gap.get("evidence_basis", [])
            if str(item).strip()
        ]

        if len(statement.split()) < 6:
            continue

        if not supporting_papers:
            continue

        if not evidence_basis:
            evidence_basis = [f"Empirical constraints identified in {supporting_papers[0]}"]

        final_verified.append({
            "gap_statement": statement,
            "supporting_papers": list(dict.fromkeys(supporting_papers)),
            "evidence_basis": evidence_basis[:8],
            "verification_reason": str(
                gap.get("verification_reason", "Verified against primary corpus evidence.")
            ).strip(),
            "verified": True,
            "support_count": len(set(supporting_papers)),
            "cross_paper_supported": len(set(supporting_papers)) >= 2,
        })

    # If the Skeptic/Verifier rejects every candidate, no gap is forced
    # through: the writer then discusses only reported limitations.

    print(
        f"   🏁 [Gap Verification Gate]: "
        f"{len(final_verified)} gaps passed final verification."
    )

    return {
        "research_gap_analysis": {
            "status": "VERIFIED",
            "proposed_gaps": grounded_candidates,
            "challenged_gaps": challenged_gaps,
            "verified_gaps": final_verified,
            "stage_usage": {
                "proposer": _usage_delta(usage_start, usage_after_proposer),
                "skeptic_verifier": _usage_delta(usage_after_proposer, snapshot_token_usage()),
            },
            "verification_policy": {
                "evidence_grounded_only": True,
                "unsupported_generalizations_rejected": True,
                "future_work_not_treated_as_proven_gap": True,
                "exact_source_titles_required": True,
            },
        },
        "verified_research_gaps": final_verified,
    }


# =====================================================================
# LEGACY COMPATIBILITY LAYER
# =====================================================================

# AGENT 6 - SUMMARY ADAPTER: reshapes the analysis into the older 'summary' format that the taxonomy,
# citation and writer code expect. No LLM is used.
# For each analysed paper it copies title, authors, year, links and joins the analysis lists
# (methodology, dataset, findings, limitations, future work) into plain text fields.
# Result goes to state.summarized_papers.
# Reshapes the analysis into a simple format (no AI).
def summarization_agent_node(
    state: ResearchState
) -> Dict[str, Any]:

    """
    Converts Section 2 structured analysis into the legacy
    summarized_papers format expected by the existing downstream
    taxonomy/citation/writing components.
    """

    print(
        "\n📋 [Summary Adapter]: "
        "Converting Section 2 analysis into downstream format..."
    )

    summaries = []

    for paper in state.analyzed_papers:

        analysis = paper.get(
            "paper_analysis",
            {}
        )

        summaries.append({

            "title": paper.get(
                "title",
                ""
            ),

            "authors": paper.get(
                "authors",
                []
            ),

            "published_year": paper.get(
                "year"
            ),

            "url": paper.get(
                "url"
            ),

            "pdf_url": paper.get(
                "pdf_url"
            ),

            "doi": paper.get(
                "doi"
            ),

            "arxiv_id": paper.get(
                "arxiv_id"
            ),

            "analysis": {

                "problem_addressed":
                    analysis.get(
                        "research_problem",
                        "Not reported."
                    ),

                "methodology":
                    "; ".join(
                        analysis.get(
                            "methodology",
                            []
                        )
                    ),

                "dataset_used":
                    "; ".join(
                        analysis.get(
                            "dataset",
                            []
                        )
                    ),

                "results":
                    "; ".join(
                        analysis.get(
                            "key_findings",
                            []
                        )
                    ),

                "limitations":
                    "; ".join(
                        analysis.get(
                            "limitations",
                            []
                        )
                    ),

                "future_work":
                    "; ".join(
                        analysis.get(
                            "future_work",
                            []
                        )
                    )
            }
        })

    print(
        f"✨ [Summary Adapter]: "
        f"Prepared {len(summaries)} structured paper records."
    )

    return {
        "summarized_papers": summaries
    }

# Backup taxonomy (no LLM): groups papers by which kind of evidence they contain.
# Categories: Methodologies and Models, Key Findings, Research Contributions, Research Limitations,
# Future Research Directions. A paper is added to a category if its analysis has something in that field.
# Used when the LLM taxonomy call fails.
# Groups the papers into themes. If the AI fails, build_deterministic_taxonomy does it with rules.
def build_deterministic_taxonomy(
    analyzed_papers: List[Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Build a safe fallback taxonomy directly from extracted
    paper-analysis evidence.

    No LLM is used here.
    No new facts are introduced.
    """

    taxonomy: Dict[str, List[str]] = {}

    keyword_groups = {
        "Methodologies and Models": [
            "methodology",
            "models_and_techniques"
        ],
        "Key Findings": [
            "key_findings"
        ],
        "Research Contributions": [
            "contributions"
        ],
        "Research Limitations": [
            "limitations"
        ],
        "Future Research Directions": [
            "future_work"
        ]
    }

    for paper in analyzed_papers:

        title = str(
            paper.get("title", "")
        ).strip()

        if not title:
            continue

        analysis = paper.get(
            "paper_analysis",
            {}
        )

        for category, fields in keyword_groups.items():

            evidence_found = False

            for field in fields:

                values = analysis.get(
                    field,
                    []
                )

                if isinstance(values, list):
                    if any(
                        str(value).strip()
                        for value in values
                    ):
                        evidence_found = True
                        break

                elif str(values).strip():
                    evidence_found = True
                    break

            if evidence_found:

                taxonomy.setdefault(
                    category,
                    []
                )

                if title not in taxonomy[category]:

                    taxonomy[category].append(
                        title
                    )

    return taxonomy

# AGENT 7 - TAXONOMY: groups the papers into 2-4 meaningful research themes.
#  1. Builds a text with each paper's problem, methods, findings, contributions and limitations.
#  2. Asks the LLM for themes in the TaxonomySchema shape (a paper may be in several themes).
#  3. If the LLM fails, uses build_deterministic_taxonomy instead.
# Result goes to state.taxonomy_map; the writer uses the themes as section headings.
def taxonomy_agent_node(
    state: ResearchState
) -> Dict[str, Any]:

    print(
        "\n🗂️ [Taxonomy Agent]: "
        "Discovering themes from analyzed literature..."
    )

    llm = make_llm("openai/gpt-oss-20b", temperature=0)

    structured_llm = llm.with_structured_output(
        TaxonomySchema
    )

    context_parts = []

    for paper in state.analyzed_papers:

        analysis = paper.get(
            "paper_analysis",
            {}
        )

        context_parts.append(
            f"""
Paper: {paper.get('title', '')}

Research Problem:
{analysis.get('research_problem', '')}

Methodology:
{analysis.get('methodology', [])}

Techniques:
{analysis.get('models_and_techniques', [])}

Key Findings:
{analysis.get('key_findings', [])}

Contributions:
{analysis.get('contributions', [])}

Limitations:
{analysis.get('limitations', [])}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
Research Topic:
{state.topic}

Analyze the following structured evidence from the selected
research papers.

Identify meaningful academic themes/categories.

Rules:

1. Use only the supplied evidence.
2. Do not invent themes unrelated to the papers.
3. A category should represent a meaningful research theme.
4. Use exact paper titles.
5. A paper may belong to multiple categories.
6. Avoid creating one category per paper.

PAPER EVIDENCE:

{context}
"""

    try:

        structured_taxonomy = structured_llm.invoke(
            prompt
        )

        taxonomy = {
            category.category_name:
                category.paper_titles
            for category
            in structured_taxonomy.categories
        }

        print(
            f"✅ [Taxonomy Agent]: "
            f"Discovered {len(taxonomy)} themes."
        )

        return {
            "taxonomy_map": taxonomy
        }

    except Exception as error:
        print(
            f"⚠️ [Taxonomy Agent]: "
            f"{error}"
        )

        print(
            "↪ Using deterministic taxonomy fallback..."
        )

        fallback_taxonomy = build_deterministic_taxonomy(
            state.analyzed_papers
        )

        print(
            f"✅ [Taxonomy Agent]: "
            f"Fallback discovered "
            f"{len(fallback_taxonomy)} themes."
        )

        return {
            "taxonomy_map": fallback_taxonomy
        }

# AGENT 8 - CITATIONS: builds the numbered reference list.
# For each paper it prepares the author text (first author + 'et al.' when there are many),
# the year, and DOI/arXiv/URL, then writes an IEEE string like  [1] Authors, "Title," Year. doi: ...
# and an APA 7 string. The numbers [1], [2], ... are the same numbers the writer cites in the text.
# No LLM is used. Result goes to state.citations_list.
def citation_agent_node(state: ResearchState) -> Dict[str, Any]:
    print(
        f"\n📋 [Citation Agent Node]: "
        f"Generating structured references for "
        f"{len(state.summarized_papers)} papers..."
    )

    citations = []

    for index, p in enumerate(state.summarized_papers, start=1):

        title = p.get("title", "").strip()
        authors = p.get("authors", [])
        year = p.get("published_year")
        url = p.get("url")
        doi = p.get("doi")
        arxiv_id = p.get("arxiv_id")

        print(
            f"✍️ Formatting Citations For [{index}]: "
            f"{title[:50]}..."
        )

        # ---------------------------------------------------------
        # Prepare author string
        # ---------------------------------------------------------

        if isinstance(authors, list):
            cleaned_authors = [str(a).strip() for a in authors if str(a).strip()]
            if len(cleaned_authors) > 3:
                author_text = f"{cleaned_authors[0]} et al."
            elif cleaned_authors:
                author_text = ", ".join(cleaned_authors)
            else:
                author_text = "Unknown Author"
        else:
            author_text = str(authors or "").strip() or "Unknown Author"

        year_text = (
            str(year)
            if year
            else "n.d."
        )

        # ---------------------------------------------------------
        # Deterministic IEEE citation (clean [index] prefix)
        # ---------------------------------------------------------

        ieee_parts = [
            f"[{index}]",
            author_text + ",",
            f'"{title},"',
            f"{year_text}."
        ]

        if doi:
            clean_doi = str(doi).replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
            ieee_parts.append(f"doi: {clean_doi}.")
        elif arxiv_id:
            ieee_parts.append(f"arXiv:{arxiv_id}.")
        elif url:
            ieee_parts.append(url)

        ieee = " ".join(ieee_parts)

        # ---------------------------------------------------------
        # Deterministic APA 7 citation
        # ---------------------------------------------------------

        apa_parts = [
            f"[{index}]",
            f"{author_text}.",
            f"({year_text}).",
            f"{title}."
        ]

        if doi:
            clean_doi = str(doi).replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
            apa_parts.append(f"https://doi.org/{clean_doi}")
        elif url:
            apa_parts.append(url)

        apa = " ".join(apa_parts)

        citations.append({
            "index": index,
            "label": f"[{index}]",
            "title": title,
            "authors": author_text,
            "year": year_text,
            "ieee": ieee,
            "apa": apa
        })

    print(
        f"✅ [Citation Agent Node]: "
        f"Generated {len(citations)} numbered citations."
    )

    return {
        "citations_list": citations
    }

WRITER_MODEL = os.environ.get("MALRG_WRITER_MODEL", "openai/gpt-oss-120b")
WRITER_MAX_TOKENS = int(os.environ.get("MALRG_WRITER_MAX_TOKENS", "8000"))
WRITER_REASONING = os.environ.get("MALRG_WRITER_REASONING", "medium")


WRITER_PROMPT_BUDGET = int(os.environ.get("MALRG_WRITER_PROMPT_BUDGET", "4800"))


# Makes the per-paper evidence smaller for the writer prompt when the token budget is tight.
# At higher levels it keeps fewer items in each list, cuts each text shorter and shortens the research problem.
# Shortens each paper’s evidence so the writer prompt fits.
def _shrink_paper_evidence(papers: List[Dict[str, Any]], level: int) -> List[Dict[str, Any]]:
    """Fewer, shorter evidence items per paper for tight token budgets."""
    scale = _DETAIL_SCALES[min(level, len(_DETAIL_SCALES) - 1)]
    shrunk = []
    for paper in papers:
        item = dict(paper)
        for key, value in paper.items():
            if isinstance(value, list):
                keep = max(1, int(round(len(value) * scale))) if value else 0
                item[key] = [str(v)[: max(120, int(400 * scale))] for v in value[:keep]]
        item["research_problem"] = str(paper.get("research_problem", ""))[: max(200, int(1500 * scale))]
        shrunk.append(item)
    return shrunk


# Cleans the review text after the LLM wrote it (also used after the Reviser).
#  1. Removes duplicate brackets such as [1] [1] or [2][2].
#  2. Removes list numbers before citations such as '1. [1]'.
#  3. Deletes any comparison-matrix table the model added by mistake (the writer prompt forbids tables).
#  4. Makes sure a '## References' section exists with all references; adds missing ones at the end.
# Cleans the review: removes duplicate brackets and tables, and makes sure the References section is complete.
def postprocess_review(content: str, citations_formatted_list: List[str]) -> str:
    """Deterministic clean-up shared by the writer and the feedback reviser."""
    # 1. Remove duplicate brackets like [1] [1] or [2][2]
    content = re.sub(r'\[(\d+)\]\s*\[\1\]', r'[\1]', content)
    # 2. Remove numbered duplicate prefix like 1. [1]
    content = re.sub(r'(?m)^\s*\d+\.\s*\[(\d+)\]', r'[\1]', content)
    # 3. Strip any accidentally generated comparison matrix table
    content = re.sub(r'(?is)###\s*(?:Methodology\s+)?(?:Cross-Paper\s+)?Comparison\s+Matrix.*?(?=(?:\n##\s)|\Z)', '', content)
    content = re.sub(r'(?is)\|\s*\*\*#Paper.*?(?=(?:\n##\s)|\Z)', '', content)

    # 4. Verify all references are present in ## References
    if "## References" not in content:
        content = content.rstrip() + "\n\n## References\n\n" + "\n\n".join(citations_formatted_list)
    else:
        ref_text = content[content.find("## References"):]
        missing_citations = [
            citation_entry
            for i, citation_entry in enumerate(citations_formatted_list, start=1)
            if f"[{i}]" not in ref_text
        ]
        if missing_citations:
            content = content.rstrip() + "\n\n" + "\n\n".join(missing_citations)
    return content


# AGENT 9 - WRITER: writes the literature review.
#  1. Builds the numbered paper directory [1]..[N] and compact evidence for each paper.
#  2. Adds the cross-paper analysis, taxonomy themes and the VERIFIED gaps (only gaps with verified=True).
#  3. Builds the prompt with strict rules: cite every paper, end every factual sentence with its citation,
#     never invent numbers or authors, no tables, 1000-1400 words, fixed section layout.
#  4. Shrinks the prompt if it is too long for the per-minute token limit.
#  5. Calls gpt-oss-120b (temperature 0.2), then postprocess_review() cleans the result.
# If the LLM fails, a simple template review is built from the evidence so the run still finishes.
# Result goes to state.final_review_md (and draft_review_md).
# The AI writes the review with a citation on every factual sentence.
def writing_agent_node(state: ResearchState) -> Dict[str, Any]:
    print(
        "\n✍️ [Writing Agent Node]: "
        "Synthesizing final multi-section review document..."
    )

    # ---------------------------------------------------------
    # Build COMPACT evidence with explicit [1] to [N] index
    # ---------------------------------------------------------

    compact_papers = []
    paper_directory = []

    for index, paper in enumerate(state.analyzed_papers, start=1):

        analysis = paper.get(
            "paper_analysis",
            {}
        )

        authors = paper.get("authors", [])
        if isinstance(authors, list):
            cleaned_authors = [str(a).strip() for a in authors if str(a).strip()]
            if len(cleaned_authors) > 2:
                author_str = f"{cleaned_authors[0]} et al."
            elif cleaned_authors:
                author_str = ", ".join(cleaned_authors)
            else:
                author_str = "Unknown Author"
        else:
            author_str = str(authors or "").strip() or "Unknown Author"

        title = paper.get("title", f"Study {index}").strip()
        year = paper.get("published_year") or paper.get("year") or "n.d."
        venue = paper.get("venue", "")

        paper_directory.append(
            f"[{index}] \"{title}\" ({year}) by {author_str}" + (f" in {venue}" if venue else "")
        )

        compact_papers.append({
            "paper_index": index,
            "citation_label": f"[{index}]",
            "title": title,
            "authors": author_str,
            "year": year,
            "venue": venue,
            "research_problem": analysis.get(
                "research_problem",
                ""
            ),
            "objectives": analysis.get(
                "objectives",
                []
            )[:5],
            "methodology": analysis.get(
                "methodology",
                []
            )[:6],
            "models_and_techniques": analysis.get(
                "models_and_techniques",
                []
            )[:8],
            "key_findings": analysis.get(
                "key_findings",
                []
            )[:8],
            "contributions": analysis.get(
                "contributions",
                []
            )[:6],
            "limitations": analysis.get(
                "limitations",
                []
            )[:6],
            "future_work": analysis.get(
                "future_work",
                []
            )[:6],
            "evidence": analysis.get(
                "evidence",
                []
            )[:12]
        })

    total_papers_count = len(compact_papers)
    paper_directory_text = "\n".join(paper_directory)

    # ---------------------------------------------------------
    # Compact cross-paper analysis
    # ---------------------------------------------------------

    cross_analysis = state.cross_paper_analysis

    if isinstance(cross_analysis, dict):

        compact_cross_analysis = {
            key: value
            for key, value in cross_analysis.items()
            if key not in {
                "raw_text",
                "full_text",
                "evidence",
                "paper_details"
            }
        }

    else:

        cross_text = str(
            cross_analysis or ""
        )

        compact_cross_analysis = cross_text[:12000]

    # ---------------------------------------------------------
    # Compact taxonomy
    # ---------------------------------------------------------

    taxonomy = state.taxonomy_map

    if isinstance(taxonomy, dict):

        compact_taxonomy = {}

        for key, value in taxonomy.items():

            if isinstance(value, list):
                compact_taxonomy[key] = value[:8]

            elif isinstance(value, str):
                compact_taxonomy[key] = value[:1500]

            else:
                compact_taxonomy[key] = value

    else:

        compact_taxonomy = str(
            taxonomy or ""
        )[:8000]

    # ---------------------------------------------------------
    # Verified research gaps
    # ---------------------------------------------------------

    verified_gaps = state.verified_research_gaps or []

    compact_verified_gaps = [
        {
            "gap_statement": gap.get("gap_statement", ""),
            "supporting_papers": gap.get("supporting_papers", []),
            "evidence_basis": gap.get("evidence_basis", []),
            "verification_reason": gap.get("verification_reason", ""),
            "verified": gap.get("verified", False),
        }
        for gap in verified_gaps
        if gap.get("verified") is True
    ]

    # ---------------------------------------------------------
    # Clean structured references list
    # ---------------------------------------------------------

    citations = state.citations_list
    citations_formatted_list = [
        c.get("ieee") or f"[{i}] {c.get('authors', 'Authors')}, \"{c.get('title', '')},\" {c.get('year', '')}."
        for i, c in enumerate(citations, start=1)
    ]
    citations_block = "\n\n".join(citations_formatted_list)

    # ---------------------------------------------------------
    # Final writing prompt
    # ---------------------------------------------------------

    prompt = f"""
Write a rigorous academic literature review on:

Research Topic:
{state.topic}

SUPPLIED SCHOLARLY PAPERS DIRECTORY (Total: {total_papers_count} papers):
{paper_directory_text}

CRITICAL CITATION GROUNDING RULES:
1. CITATION COMPLETENESS: Exactly {total_papers_count} papers are supplied, numbered [1] through [{total_papers_count}].
   - You MUST cite ALL {total_papers_count} papers across the thematic sections in the review.
   - Do NOT stop after citing [1], [2], [3]. Every single reference number [1] through [{total_papers_count}] must appear in the manuscript body.
   - Weave each paper's citation [i] into the thematic section where its methodology, dataset, or findings are discussed.
2. SOURCE GROUNDING:
   - Ground all discussions strictly in the supplied papers and their extracted evidence.
   - Use the REAL author names and real titles from the directory. Do NOT hallucinate authors like "Kung et al." or "Huang et al." unless that paper actually has those authors.
   - Do NOT invent general multi-agent debate or literature-review verification concepts unless the retrieved papers explicitly address that domain.
3. NO COMPARISON MATRIX TABLE:
   - Do NOT output any "Methodology Comparison Matrix" or Markdown comparison table.
   - Discuss methodologies, architectures, evaluation metrics, and empirical limitations in cohesive, thematic academic paragraphs.
4. TAXONOMY AND THEMES:
   - Group the literature into 2 to 4 distinct thematic sections based on actual technical differences.
   - Synthesize multiple studies within each theme (e.g. "[1], [4] investigate vision-language clinical models, whereas [2], [5] focus on clinical reasoning...").
5. CLEAN REFERENCES SECTION:
   - End the review with a ## References section.
   - List every single paper [1] through [{total_papers_count}] in clean numerical order.
   - Never output duplicate brackets like "[1] [1]" or "1. [1]". Each entry must start cleanly with "[i]".
6. LENGTH: the body (excluding References) should be 1000 to 1400 words.
7. SENTENCE-LEVEL ATTRIBUTION:
   - Every sentence that reports a method, dataset, result, number, or limitation must end with the
     citation(s) of the specific paper(s) it comes from, e.g. "... reached 94% accuracy [3]."
   - State only facts present in that paper's PAPER EVIDENCE; never add numbers, datasets, or claims
     that are not in the supplied evidence.
   - Comparison sentences must cite every compared paper, e.g. "[1] relies on X, whereas [4] uses Y."
   - Use calibrated wording ("reports", "suggests", "indicates"); never "proves" or "definitively".
   - In "Research Gaps and Open Directions", every gap must cite the paper(s) whose reported
     limitations or findings support it.

TAXONOMY:
{compact_taxonomy}

PAPER EVIDENCE:
{compact_papers}

CROSS-PAPER ANALYSIS:
{compact_cross_analysis}

VERIFIED RESEARCH GAPS:
{compact_verified_gaps}

REFERENCES TO INCLUDE AT THE END:
{citations_block}

OUTPUT FORMAT:
# Related Work

[Optional concise opening paragraph framing the research domain based on the supplied literature.]

## [Theme 1 derived from actual paper topics]
[Thematic synthesis weaving relevant papers among [1]-[{total_papers_count}].]

## [Theme 2 derived from actual paper topics]
[Thematic synthesis weaving relevant papers among [1]-[{total_papers_count}].]

## [Theme 3 derived from actual paper topics]
[Thematic synthesis weaving relevant papers among [1]-[{total_papers_count}].]

## Research Gaps and Open Directions
[Evidence-grounded discussion of limitations and gaps supported by the studies.]

## Conclusion
[Concise summary of findings and research directions.]

## References
{citations_block}
"""

    # Keep the prompt within the per-minute token budget (the answer needs ~3K tokens).
    current_papers = str(compact_papers)
    current_cross = str(compact_cross_analysis)
    for level in range(1, len(_DETAIL_SCALES)):
        if estimate_tokens(prompt) <= WRITER_PROMPT_BUDGET:
            break
        smaller_papers = str(_shrink_paper_evidence(compact_papers, level))
        prompt = prompt.replace(current_papers, smaller_papers)
        current_papers = smaller_papers
        if len(current_cross) > 50:
            cross_text = (
                json.dumps(compact_cross_analysis, ensure_ascii=False)
                if isinstance(compact_cross_analysis, dict)
                else str(compact_cross_analysis)
            )
            smaller_cross = cross_text[: _CROSS_CHARS[level]]
            prompt = prompt.replace(current_cross, smaller_cross)
            current_cross = smaller_cross
    print(f"   📏 [Writing Agent] prompt ≈ {estimate_tokens(prompt)} tokens")

    # ---------------------------------------------------------
    # Generate final review
    # ---------------------------------------------------------

    try:

        llm = make_llm(
            WRITER_MODEL,
            temperature=0.2,
            max_tokens=WRITER_MAX_TOKENS,
            reasoning_effort=WRITER_REASONING,
        )

        content = invoke_text(llm, prompt, label="Writing Agent")
        content = postprocess_review(content, citations_formatted_list)

        print(
            "✅ [Writing Agent Node]: "
            "Final literature review generated and validated."
        )

        return {
            "final_review_md": content,
            "draft_review_md": content,
        }

    except Exception as error:

        if isinstance(error, LLMUnavailable) or strict_mode():
            raise


        print(
            f"❌ [Writing Agent Node]: "
            f"Generation failed: {error}"
        )

        # -----------------------------------------------------
        # Safe fallback with complete citation coverage
        # -----------------------------------------------------

        fallback = (
            f"# Literature Review: {state.topic}\n\n"
            "## Thematic Advances and Research Trajectories\n\n"
            f"The reviewed corpus encompasses {total_papers_count} peer-reviewed studies "
            f"published between {state.start_year} and {state.end_year}. "
            "The following thematic synthesis integrates the reported methodologies, "
            "empirical findings, and recognized limitations across these contributions.\n\n"
        )

        # Group papers into thematic sections
        mid = max(1, total_papers_count // 2)
        group_1 = compact_papers[:mid]
        group_2 = compact_papers[mid:]

        fallback += "## Methodological Architectures and Diagnostic Paradigms\n\n"
        for p in group_1:
            idx = p["paper_index"]
            fallback += (
                f"{p['title']} [{idx}] addresses {p['research_problem'] or 'domain challenges'}. "
            )
            if p["methodology"]:
                fallback += f"The authors employ {'; '.join(p['methodology'][:3])}. "
            if p["key_findings"]:
                fallback += f"Reported empirical outcomes establish that {'; '.join(p['key_findings'][:3])}. "
            if p["limitations"]:
                fallback += f"However, notable constraints include {'; '.join(p['limitations'][:2])}. "
            fallback += "\n\n"

        if group_2:
            fallback += "## Empirical Evaluations, Generalization, and Domain Transfer\n\n"
            for p in group_2:
                idx = p["paper_index"]
                fallback += (
                    f"Similarly, {p['title']} [{idx}] examines {p['research_problem'] or 'operational requirements'}. "
                )
                if p["methodology"]:
                    fallback += f"The methodological pipeline incorporates {'; '.join(p['methodology'][:3])}. "
                if p["key_findings"]:
                    fallback += f"Key results demonstrate {'; '.join(p['key_findings'][:3])}. "
                if p["limitations"]:
                    fallback += f"Identified challenges center upon {'; '.join(p['limitations'][:2])}. "
                fallback += "\n\n"

        fallback += "## Research Gaps and Open Directions\n\n"
        if compact_verified_gaps:
            for gap in compact_verified_gaps:
                fallback += (
                    f"- **Verified Research Gap:** {gap['gap_statement']}\n"
                )
                if gap.get("evidence_basis"):
                    fallback += f"  - Evidence: {'; '.join(gap['evidence_basis'])}\n"
        else:
            fallback += (
                "Across the evaluated studies, primary open research challenges include "
                "cross-domain generalization, real-world benchmark standardization, "
                "and robust evidence-grounded attribution without error propagation.\n"
            )

        fallback += (
            "\n## Conclusion\n\n"
            f"In summary, the {total_papers_count} examined studies present diverse "
            "methodological trajectories ranging from specialized model architectures "
            "to multi-institutional clinical evaluation frameworks. Addressing the "
            "identified evidentiary gaps represents a crucial imperative for future research.\n\n"
            "## References\n\n"
            f"{citations_block}\n"
        )

        return {
            "final_review_md": fallback
        }

# Returns the list of ready-made reference strings (the 'ieee' text) from the citation records.
# Used by postprocess_review and the feedback loop to rebuild the References section.
def _formatted_citations(citations: List[Dict[str, Any]]) -> List[str]:
    return [
        c.get("ieee") or f"[{i}] {c.get('authors', 'Authors')}, \"{c.get('title', '')},\" {c.get('year', '')}."
        for i, c in enumerate(citations, start=1)
    ]


# AGENT 10 - FEEDBACK (only in the proposed mode M5): the system checks and improves its own review.
# It calls run_feedback_loop(): critic score -> fact-check -> Reviser rewrite -> re-score, for up to
# feedback_rounds rounds (default 2). A revision is kept only if the score improves.
# In other modes it does nothing. Result: improved final_review_md plus feedback_log.
def feedback_agent_node(state: ResearchState) -> Dict[str, Any]:
    """
    Critic -> Fact-Checker -> Reviser loop (proposed system only).
    Revisions are kept only when the critic score improves.
    """
    if state.ablation_mode != "M5_full_feedback":
        return {"feedback_log": []}
    if not state.final_review_md or not state.analyzed_papers:
        return {"feedback_log": []}

    print("\n🔁 [Feedback Agent]: Critiquing and revising the draft review...")
    citations = _formatted_citations(state.citations_list)

    try:
        result = run_feedback_loop(
            state.final_review_md,
            state.analyzed_papers,
            postprocess=lambda text: postprocess_review(text, citations),
            max_rounds=state.feedback_rounds,
        )
    except Exception as error:
        if isinstance(error, LLMUnavailable) or strict_mode():
            raise
        print(f"   ⚠️ [Feedback Agent]: feedback loop skipped ({error}); keeping the draft.")
        return {"feedback_log": [{"round": 0, "error": str(error)}]}

    return {
        "final_review_md": result["final_review_md"],
        "feedback_log": result["feedback_log"],
    }


# =====================================================================
# AGENT GRAPH COMPILATION
# =====================================================================
workflow = StateGraph(ResearchState)
workflow.add_node("search_agent", search_agent_node)
workflow.add_node("summarization_agent", summarization_agent_node)
workflow.add_node("taxonomy_agent", taxonomy_agent_node)
workflow.add_node("citation_agent", citation_agent_node)
workflow.add_node("writing_agent", writing_agent_node)
workflow.add_node("paper_extraction_agent", paper_extraction_agent_node)
workflow.add_node("paper_analysis_agent", paper_analysis_agent_node)
workflow.add_node("cross_paper_analysis_agent", cross_paper_analysis_agent_node)
workflow.add_node("research_gap_agent", research_gap_agent_node)
workflow.add_node("feedback_agent", feedback_agent_node)

workflow.add_edge(START, "search_agent")
workflow.add_edge("search_agent", "paper_extraction_agent")
workflow.add_edge("paper_extraction_agent", "paper_analysis_agent")
workflow.add_edge("paper_analysis_agent", "cross_paper_analysis_agent")
workflow.add_edge("cross_paper_analysis_agent", "research_gap_agent")
workflow.add_edge("research_gap_agent", "summarization_agent")
workflow.add_edge("summarization_agent", "taxonomy_agent")
workflow.add_edge("taxonomy_agent", "citation_agent")
workflow.add_edge("citation_agent", "writing_agent")
workflow.add_edge("writing_agent", "feedback_agent")
workflow.add_edge("feedback_agent", END)

compiled_graph = workflow.compile()

# =====================================================================
# 3. DYNAMIC EXPERIMENT STORAGE & REVIEW EVALUATION
# =====================================================================

EXPERIMENTS_DIR = Path(__file__).resolve().parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


# Converts any Python object into something that can be saved as JSON.
# Pydantic models become dicts, dicts/lists/tuples are converted element by element,
# simple values stay as they are, and anything else is turned into text with str().
def _json_safe(value):
    # Convert common runtime objects into JSON-serializable structures.
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# Saves data into a JSON file (pretty printed, UTF-8). Data is made JSON-safe first.
def _write_json(path: Path, payload):
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# Builds the folder name for one run inside experiments/:
# date_time + the topic in safe lowercase letters (max 70 chars) + the first 8 characters of the task id.
# Example: 20261009_115715_retrieval_augmented_generation_9fe437fb
def _safe_experiment_name(topic: str, task_id: str) -> str:
    topic_slug = re.sub(r"[^a-zA-Z0-9]+", "_", topic.strip()).strip("_")
    topic_slug = (topic_slug or "untitled_topic")[:70].lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{topic_slug}_{task_id[:8]}"


# Counts the words in one section of the review text.
# It finds the heading, ends at the next heading from the given list, and counts the word-like pieces in between.
def _section_words(review: str, heading: str, next_headings: List[str]) -> int:
    start = review.find(heading)
    if start == -1:
        return 0
    start += len(heading)
    end = len(review)
    for next_heading in next_headings:
        position = review.find(next_heading, start)
        if position != -1:
            end = min(end, position)
    return len(re.findall(r"\b[\w'-]+\b", review[start:end]))


# Rule-based quality check of a finished review (no LLM, fast and repeatable).
# It looks at: document structure and headings, which of the run's papers are cited and referenced,
# thematic subsections and synthesis words, a research-gap section, truncation (did the text stop mid-sentence),
# redundancy, and old rigid-template headings.
# These checks are combined into overall_quality_score out of 100 plus many detail fields.
# (Different from the NLI/LLM-judge metrics in evaluation/metrics.py, which check claims against sources.)
def evaluate_generated_review(review: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic evaluator for research-paper-style Related Work /
    Literature Review output.

    V2 evaluates thematic organization rather than requiring a rigid
    report template.
    """
    review = review or ""
    review_lower = review.lower()

    # ---------------------------------------------------------
    # Basic document structure
    # ---------------------------------------------------------
    word_count = len(re.findall(r"\b[\w'-]+\b", review))

    heading_matches = re.findall(
        r"(?m)^#{1,6}\s+(.+?)\s*$",
        review
    )

    headings = [
        h.strip()
        for h in heading_matches
        if h.strip()
    ]

    references_heading = next(
        (
            h for h in headings
            if h.lower().strip() == "references"
        ),
        None
    )

    references_index = -1
    reference_match = re.search(
        r"(?im)^#{1,6}\s+references\s*$",
        review
    )

    if reference_match:
        references_index = reference_match.start()
        references_text = review[reference_match.end():]
    else:
        references_text = ""

    reference_lines = [
        line.strip()
        for line in references_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    reference_count = len([
        line for line in reference_lines
        if len(line) >= 20
    ])

    # ---------------------------------------------------------
    # Actual papers from this run
    # ---------------------------------------------------------
    papers = (
        state.get("analyzed_papers")
        or state.get("raw_papers_list")
        or []
    )

    source_names = [
        str(paper.get("title", "")).strip()
        for paper in papers
        if str(paper.get("title", "")).strip()
    ]

    # ---------------------------------------------------------
    # Source coverage
    # ---------------------------------------------------------
    matched_sources = []

    for paper in papers:
        title = str(paper.get("title", "")).strip()
        authors = paper.get("authors", [])
        candidates = []

        if isinstance(authors, list):
            for author in authors[:3]:
                name = str(author).strip()
                if name:
                    surname = name.split()[-1].rstrip(".,")
                    if len(surname) >= 3:
                        candidates.append(surname)

        matched = any(
            re.search(
                rf"\b{re.escape(candidate)}\b",
                review,
                re.I
            )
            for candidate in candidates
        )

        if not matched and title:
            tokens = [
                token.lower()
                for token in re.findall(
                    r"\b[a-zA-Z]{4,}\b",
                    title
                )
            ]

            tokens = list(dict.fromkeys(tokens))[:8]

            overlap = sum(
                token in review_lower
                for token in tokens
            )

            matched = (
                bool(tokens)
                and overlap >= max(
                    2,
                    min(4, len(tokens))
                )
            )

        if matched:
            matched_sources.append(title)

    reference_source_coverage = 0

    for title in source_names:
        tokens = [
            token.lower()
            for token in re.findall(
                r"\b[a-zA-Z]{4,}\b",
                title
            )
        ]

        tokens = list(dict.fromkeys(tokens))[:8]

        overlap = sum(
            token in review_lower
            for token in tokens
        )

        if (
            tokens
            and overlap >= max(
                2,
                min(4, len(tokens))
            )
        ):
            reference_source_coverage += 1

    # ---------------------------------------------------------
    # Thematic / synthesis indicators
    # ---------------------------------------------------------
    synthesis_terms = [
        "compared",
        "comparison",
        "whereas",
        "in contrast",
        "collectively",
        "synthesis",
        "across the studies",
        "across studies",
        "cross-paper",
        "similarly",
        "difference",
        "differences",
        "however",
        "while",
        "trend",
        "approaches",
        "studies",
        "literature"
    ]

    synthesis_indicator_count = sum(
        review_lower.count(term)
        for term in synthesis_terms
    )

    gap_terms = [
        "gap",
        "limitation",
        "limitations",
        "future work",
        "future research",
        "challenge",
        "open problem",
        "research direction",
        "unresolved",
        "remain",
        "remains"
    ]

    gap_indicator_count = sum(
        review_lower.count(term)
        for term in gap_terms
    )

    # ---------------------------------------------------------
    # Detect old rigid report-template headings
    # ---------------------------------------------------------
    legacy_headings = {
        "introduction",
        "existing research",
        "comparative analysis",
        "research gaps & limitations",
        "research gaps and limitations",
        "conclusion"
    }

    normalized_headings = {
        re.sub(r"\s+", " ", h.lower()).strip()
        for h in headings
    }

    legacy_heading_count = len(
        normalized_headings.intersection(
            legacy_headings
        )
    )

    # ---------------------------------------------------------
    # Thematic heading analysis
    # ---------------------------------------------------------
    non_reference_h2 = [
        h for h in headings
        if h.lower().strip() != "references"
        and not h.lower().strip().startswith("literature review")
        and not h.lower().strip().startswith("related work")
    ]

    thematic_headings = [
        h for h in non_reference_h2
        if h.lower().strip() not in legacy_headings
    ]

    # A Related Work should normally have at least two thematic
    # subsections, unless the literature is extremely small.
    expected_thematic_sections = (
        len(source_names) >= 3
        and len(thematic_headings) >= 2
    ) or (
        len(source_names) < 3
        and len(thematic_headings) >= 1
    )

    thematic_structure = (
        expected_thematic_sections
        and legacy_heading_count <= 2
    )

    # ---------------------------------------------------------
    # Research-gap section detection
    # ---------------------------------------------------------
    gap_section_words = 0

    gap_heading_match = re.search(
        r"(?im)^#{1,6}\s+.*(?:gap|limitation|future|open direction|open problem).*?$",
        review
    )

    if gap_heading_match:
        gap_start = gap_heading_match.end()

        next_heading = re.search(
            r"(?m)^#{1,6}\s+.+?$",
            review[gap_start:]
        )

        gap_end = (
            gap_start + next_heading.start()
            if next_heading
            else (
                references_index
                if references_index != -1
                else len(review)
            )
        )

        gap_section_words = len(
            re.findall(
                r"\b[\w'-]+\b",
                review[gap_start:gap_end]
            )
        )

    # ---------------------------------------------------------
    # Truncation checks
    # ---------------------------------------------------------
    stripped = review.rstrip()
    truncation_flags = []

    if not stripped:
        truncation_flags.append("empty_review")

    if references_heading is None:
        truncation_flags.append("missing_references")

    if stripped.endswith(
        (",", ":", ";", "-", "(", "[", "{")
    ):
        truncation_flags.append(
            "ends_with_incomplete_punctuation"
        )

    if stripped.count("```") % 2 != 0:
        truncation_flags.append(
            "unclosed_code_fence"
        )

    # ---------------------------------------------------------
    # Redundancy
    # ---------------------------------------------------------
    sentences = [
        re.sub(
            r"\s+",
            " ",
            sentence
        ).strip().lower()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            review
        )
        if len(sentence.split()) >= 8
    ]

    sentence_counts = Counter(sentences)

    repeated_sentences = sum(
        count - 1
        for count in sentence_counts.values()
        if count > 1
    )

    redundancy_ratio = (
        repeated_sentences / len(sentences)
        if sentences
        else 0.0
    )

    # ---------------------------------------------------------
    # Checks
    # ---------------------------------------------------------
    checks = {
        "structure": (
            thematic_structure
            and references_heading is not None
        ),

        "word_count": (
            900 <= word_count <= 1600
        ),

        "references": (
            reference_count >= len(source_names)
            if source_names
            else reference_count > 0
        ),

        "source_coverage": (
            reference_source_coverage >= len(source_names)
            if source_names
            else True
        ),

        "synthesis": (
            synthesis_indicator_count >= 4
            and len(thematic_headings) >= (
                2 if len(source_names) >= 3 else 1
            )
        ),

        "research_gaps": (
            gap_indicator_count >= 2
            and (
                gap_section_words >= 40
                or gap_indicator_count >= 4
            )
        ),

        "completeness": (
            references_heading is not None
            and reference_count > 0
            and word_count >= 900
        ),

        "truncation": not truncation_flags,

        "redundancy": (
            redundancy_ratio <= 0.10
        ),
    }

    # ---------------------------------------------------------
    # Penalize clearly rigid old-template output
    # ---------------------------------------------------------
    if legacy_heading_count >= 5:
        checks["structure"] = False
        checks["synthesis"] = False

    # ---------------------------------------------------------
    # Score
    # ---------------------------------------------------------
    weights = {
        "structure": 20,
        "word_count": 10,
        "references": 10,
        "source_coverage": 15,
        "synthesis": 15,
        "research_gaps": 10,
        "completeness": 10,
        "truncation": 5,
        "redundancy": 5,
    }

    score = round(
        100
        * sum(
            weights[key]
            for key, passed in checks.items()
            if passed
        )
        / sum(weights.values()),
        2
    )

    # ---------------------------------------------------------
    # Return backward-compatible evaluation structure
    # ---------------------------------------------------------
    section_word_counts = {
        "Related Work": word_count,
        "Thematic Sections": len(thematic_headings),
        "Research Gaps": gap_section_words,
        "References": len(
            re.findall(
                r"\b[\w'-]+\b",
                references_text
            )
        )
    }

    return {
        "evaluation_version": "related-work-v2",

        "evaluated_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "word_count": word_count,

        "word_count_target": "900-1600",

        "structure": {
            "required_headings": [
                "Thematic subsections",
                "References"
            ],
            "present": {
                "thematic_sections": thematic_structure,
                "references": references_heading is not None
            },
            "thematic_headings": thematic_headings,
            "legacy_heading_count": legacy_heading_count,
            "pass": checks["structure"]
        },

        "references": {
            "reference_count": reference_count,
            "expected_source_count": len(source_names),
            "pass": checks["references"]
        },

        "source_coverage": {
            "source_titles": source_names,
            "matched_sources": matched_sources,
            "matched_source_count": len(matched_sources),
            "reference_source_coverage_count": (
                reference_source_coverage
            ),
            "pass": checks["source_coverage"]
        },

        "synthesis": {
            "comparative_section_words": (
                word_count
            ),
            "indicator_count": (
                synthesis_indicator_count
            ),
            "thematic_section_count": len(
                thematic_headings
            ),
            "pass": checks["synthesis"]
        },

        "research_gaps": {
            "section_words": gap_section_words,
            "indicator_count": gap_indicator_count,
            "pass": checks["research_gaps"]
        },

        "section_word_counts": section_word_counts,

        "completeness": {
            "conclusion_words": 0,
            "references_words": section_word_counts[
                "References"
            ],
            "pass": checks["completeness"]
        },

        "truncation": {
            "flags": truncation_flags,
            "pass": checks["truncation"]
        },

        "redundancy": {
            "sentence_count": len(sentences),
            "repeated_sentence_count": repeated_sentences,
            "ratio": round(
                redundancy_ratio,
                4
            ),
            "pass": checks["redundancy"]
        },

        "checks": checks,

        "overall_quality_score": score
    }

# Saves everything about a finished run into its own folder under experiments/.
# Files written: request.json, search_results.json, extracted_papers.json, analyzed_papers.json,
# cross_paper_analysis.json, research_gap_analysis.json, verified_research_gaps.json, taxonomy.json,
# citations.json, feedback_log.json, draft_review.md, final_review.md, evaluation.json, run_state.json,
# experiment_summary.json. Returns the folder path. This is the permanent record used for analysis and reports.
def persist_completed_experiment(
    task_id: str,
    topic: str,
    max_papers: int,
    final_state: Dict[str, Any],
) -> str:
    # Persist the complete, actual output of one generated experiment.
    experiment_dir = EXPERIMENTS_DIR / _safe_experiment_name(topic, task_id)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    state = _json_safe(final_state)
    review = str(state.get("final_review_md", "") or "")
    evaluation = evaluate_generated_review(review, state)

    _write_json(
        experiment_dir / "request.json",
        {
            "task_id": task_id,
            "topic": topic,
            "max_papers": max_papers,
            "ablation_mode": state.get("ablation_mode"),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    _write_json(
        experiment_dir / "search_results.json",
        state.get("raw_papers_list", []),
    )
    _write_json(
        experiment_dir / "extracted_papers.json",
        state.get("extracted_papers", []),
    )
    _write_json(
        experiment_dir / "analyzed_papers.json",
        state.get("analyzed_papers", []),
    )
    _write_json(
        experiment_dir / "cross_paper_analysis.json",
        state.get("cross_paper_analysis", {}),
    )
    _write_json(
        experiment_dir / "research_gap_analysis.json",
        state.get("research_gap_analysis", {}),
    )
    _write_json(
        experiment_dir / "verified_research_gaps.json",
        state.get("verified_research_gaps", []),
    )
    _write_json(
        experiment_dir / "taxonomy.json",
        state.get("taxonomy_map", {}),
    )
    _write_json(
        experiment_dir / "citations.json",
        state.get("citations_list", []),
    )
    _write_json(
        experiment_dir / "feedback_log.json",
        state.get("feedback_log", []),
    )
    if state.get("draft_review_md"):
        (experiment_dir / "draft_review.md").write_text(
            str(state.get("draft_review_md")), encoding="utf-8"
        )
    (experiment_dir / "final_review.md").write_text(review, encoding="utf-8")
    _write_json(experiment_dir / "evaluation.json", evaluation)
    _write_json(experiment_dir / "run_state.json", state)
    _write_json(
        experiment_dir / "experiment_summary.json",
        {
            "task_id": task_id,
            "topic": topic,
            "max_papers": max_papers,
            "actual_papers_selected": len(state.get("raw_papers_list", [])),
            "actual_papers_analyzed": len(state.get("analyzed_papers", [])),
            "verified_research_gap_count": len(
                state.get("verified_research_gaps", [])
            ),
            "word_count": evaluation["word_count"],
            "overall_quality_score": evaluation["overall_quality_score"],
            "checks": evaluation["checks"],
            "experiment_directory": str(experiment_dir),
        },
    )

    return str(experiment_dir)


# =====================================================================
# 4. ASYNCHRONOUS BACKEND BACKGROUND TASK LOGIC
# =====================================================================
# Runs the whole 10-agent pipeline in the background (started by trigger_generation).
#  1. Builds initial_input from topic, years, max_papers, mode.
#  2. Streams the LangGraph: after every agent finishes it adds {agent, time, seconds} to progress,
#     which the web page shows as green check marks.
#  3. Saves the run to experiments/ and scores the final review.
#  4. Stores status COMPLETED with the data in TASKS_DB.
#  5. On an error it stores RATE_LIMITED (Groq quota) or FAILED with the message.
def run_agent_workflow_async(
    task_id: str,
    topic: str,
    start_year: int,
    end_year: int,
    max_papers: int,
    ablation_mode: str = DEFAULT_ABLATION_MODE,
):
    # Execute the graph and automatically store the actual run artifacts.
    try:
        if ablation_mode not in ABLATION_MODES:
            ablation_mode = DEFAULT_ABLATION_MODE
        initial_input = {
            "topic": topic,
            "start_year": start_year,
            "end_year": end_year,
            "max_papers": max_papers,
            "ablation_mode": ablation_mode,
            "feedback_rounds": FEEDBACK_MAX_ROUNDS,
        }
        # Stream the graph so the API can report which agent has finished and how long it took.
        progress: List[Dict[str, Any]] = []
        TASKS_DB[task_id]["progress"] = progress
        final_state: Dict[str, Any] = {}
        step_started = time.perf_counter()
        for mode, chunk in compiled_graph.stream(initial_input, stream_mode=["updates", "values"]):
            if mode == "values":
                final_state = chunk
                continue
            now = time.perf_counter()
            for node_name in chunk:
                progress.append({
                    "agent": node_name,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                    "seconds": round(now - step_started, 2),
                })
            step_started = now
        final_state = _json_safe(final_state)

        experiment_dir = persist_completed_experiment(
            task_id=task_id,
            topic=topic,
            max_papers=max_papers,
            final_state=final_state,
        )

        evaluation = evaluate_generated_review(
            str(final_state.get("final_review_md", "") or ""),
            final_state,
        )

        TASKS_DB[task_id] = {
            "status": "COMPLETED",
            "topic": topic,
            "data": final_state,
            "evaluation": evaluation,
            "experiment_dir": experiment_dir,
            "progress": progress,
        }

        print(
            f"📦 [Backend Task Queue]: Task '{task_id}' processed successfully."
        )
        print(f"🧪 [Experiment Logger]: Saved run to '{experiment_dir}'")
        print(
            f"📊 [Dynamic Evaluator]: Overall quality = "
            f"{evaluation['overall_quality_score']}/100"
        )

    except Exception as e:
        error_text = str(e)
        lowered = error_text.lower()

        if (
            "rate limit" in lowered
            or "tokens per day" in lowered
            or "tpd" in lowered
            or "quota" in lowered
        ):
            TASKS_DB[task_id] = {
                "status": "RATE_LIMITED",
                "topic": topic,
                "data": None,
                "error": error_text,
                "reason": (
                    "Groq token quota or rate limit was reached. "
                    "The literature review generation was halted."
                ),
                "retryable": True,
            }

            print(
                f"🛑 [Backend Task Queue]: Task '{task_id}' "
                f"halted because the Groq quota/rate limit was reached."
            )
        else:
            TASKS_DB[task_id] = {
                "status": "FAILED",
                "topic": topic,
                "error": error_text,
                "reason": error_text,
                "retryable": False,
            }

            print(
                f"❌ [Backend Task Queue]: Task '{task_id}' crashed. "
                f"Error: {error_text}"
            )

# =====================================================================
# 4. WEB REST ENDPOINTS (API DISPATCH ROUTERS)
# =====================================================================

# API endpoint POST /api/review/generate (Python side): starts a new review.
# It makes a task_id, stores {status: PROCESSING} in TASKS_DB, schedules run_agent_workflow_async as a
# background task and immediately returns the task_id. The Node server later polls the status.
@app.post("/api/review/generate")
async def trigger_generation(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Endpoint A: Triggers the multi-agent generation graph. 
    Acknowledges requests immediately by spawning a background worker.
    """
    # Generate a unique task descriptor hash UUID
    task_id = str(uuid.uuid4())
    
    # Initialize the memory tracking slot configuration
    TASKS_DB[task_id] = {
        "status": "PROCESSING",
        "topic": request.topic,
        "data": None
    }
    
    # Add the heavy execution function to FastAPI's background thread task manager
    background_tasks.add_task(
        run_agent_workflow_async, 
        task_id=task_id, 
        topic=request.topic, 
        start_year=request.start_year,
        end_year=request.end_year,
        max_papers=request.max_papers,
        ablation_mode=request.ablation_mode,
    )
    
    # Return instantaneous token handshake acknowledgment status back to client browser
    return {
        "message": "Literature review orchestration worker started successfully.",
        "task_id": task_id,
        "status": "PROCESSING"
    }




# API endpoint GET /api/review/status/{task_id}: the polling target.
# Returns the task's current record from TASKS_DB (status, progress list, and the data when completed).
# 404 if the task id is unknown (for example after a restart of the Python server).
@app.get("/api/review/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Endpoint B: Polling target. Allows the frontend to request updates 
    for a specific running graph worker execution using its unique task ID.
    """
    if task_id not in TASKS_DB:
        raise HTTPException(status_code=404, detail="Requested generation transaction ID not found.")
        
    return TASKS_DB[task_id]


# API endpoint: returns the quality scores (evaluation) of a finished task.
# If the task is not COMPLETED yet it returns just its status and evaluation=None.
@app.get("/api/review/evaluation/{task_id}")
async def get_task_evaluation(task_id: str):
    # Return dynamic evaluation metrics for a completed run.
    if task_id not in TASKS_DB:
        raise HTTPException(
            status_code=404,
            detail="Requested generation transaction ID not found.",
        )

    task = TASKS_DB[task_id]

    if task.get("status") != "COMPLETED":
        return {
            "status": task.get("status"),
            "evaluation": None,
        }

    return {
        "status": "COMPLETED",
        "task_id": task_id,
        "topic": task.get("topic"),
        "evaluation": task.get("evaluation"),
        "experiment_dir": task.get("experiment_dir"),
    }


# API endpoint POST /api/review/feedback/{task_id}: human-in-the-loop revision.
#  1. Finds the completed task and its review, papers and citations.
#  2. Runs revise_with_user_feedback(): the reviewer's comment plus the automatic critic findings go to the Reviser.
#  3. Replaces the stored review with the revised one, re-evaluates it and records the critic scores before/after.
#  4. Saves final_review_user_revN.md in the run folder and returns the new review and scores.
# Note: unlike the automatic loop, this path does not reject a revision that lowers the score.
@app.post("/api/review/feedback/{task_id}")
async def apply_user_feedback(task_id: str, request: UserFeedbackRequest):
    """
    Human-in-the-loop feedback: the reviewer's comments are combined with the
    automatic Critic/Fact-Checker findings and applied by the Reviser agent.
    """
    task = TASKS_DB.get(task_id)
    if not task or task.get("status") != "COMPLETED" or not task.get("data"):
        raise HTTPException(status_code=404, detail="No completed review found for this task.")

    import asyncio

    state = task["data"]
    review = str(state.get("final_review_md", "") or "")
    citations = _formatted_citations(state.get("citations_list", []))

    try:
        result = await asyncio.to_thread(
            revise_with_user_feedback,
            review,
            request.comments,
            state.get("analyzed_papers", []),
            lambda text: postprocess_review(text, citations),
        )
    except LLMUnavailable as error:
        raise HTTPException(status_code=429, detail=f"LLM quota/rate limit: {error}")

    history = state.setdefault("user_feedback_history", [])
    history.append({
        "comments": request.comments,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "critic_score_before": result["critic_before"]["score"],
        "critic_score_after": result["critic_after"]["score"],
    })
    state["final_review_md"] = result["final_review_md"]
    evaluation = evaluate_generated_review(result["final_review_md"], state)
    task["evaluation"] = evaluation

    experiment_dir = task.get("experiment_dir")
    if experiment_dir:
        revision_no = len(history)
        Path(experiment_dir, f"final_review_user_rev{revision_no}.md").write_text(
            result["final_review_md"], encoding="utf-8"
        )
        _write_json(Path(experiment_dir) / "user_feedback_history.json", history)

    return {
        "task_id": task_id,
        "final_review_md": result["final_review_md"],
        "critic_before": result["critic_before"],
        "critic_after": result["critic_after"],
        "evaluation": evaluation,
        "history": history,
    }


# Startup script executor block
if __name__ == "__main__":
    import uvicorn
    print("[STARTUP] Starting FastAPI server router dashboard layer...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)