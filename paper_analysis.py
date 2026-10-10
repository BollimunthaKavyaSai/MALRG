import os
import json
import re
import time
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
from langchain_groq import ChatGroq
from llm_client import (
    is_daily_quota_error,
    _requested_and_limit,
    is_request_too_large,
    DailyQuotaExhausted,
    make_llm,
    retry_after_seconds,
    strict_mode,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"

MAX_CHUNK_CHARS = 14000

OVERLAP_CHARS = 1200

MAX_CHUNKS = 5

# Number of chunks selected from different parts of a paper.
MAX_EVIDENCE_CHUNKS = 5

# Characters used when selecting representative chunks.
SECTION_CONTEXT_CHARS = 14000


# ============================================================
# RATE-LIMIT SAFE LLM INVOCATION
# ============================================================

# Special error: Groq's daily token quota is used up (or limits persist in strict mode).
# It is passed upward untouched so the whole run stops cleanly and shows 'quota reached' in the UI.
class GroqQuotaExceeded(DailyQuotaExhausted):
    """Raised when Groq reports an exhausted daily/token quota."""


# Calls the LLM safely in the Paper Analysis agent.
#  - If the request is too large, it lowers max_tokens so it fits Groq's per-minute limit and retries.
#  - If the DAILY quota is finished, it raises GroqQuotaExceeded immediately (no useless retries).
#  - If it is only a per-minute rate limit, it waits (5 s, 10 s, 20 s ... up to 90 s) and tries again.
#  - Any other error is raised normally.
def invoke_with_rate_limit_retry(
    llm: ChatGroq,
    prompt: str,
    max_retries: int = 5,
    base_wait: float = 5.0
):
    """
    Invoke Groq safely.

    Temporary TPM/rate-limit errors may be retried.
    Daily token/quota exhaustion is never retried and is propagated
    immediately so the complete generation can halt cleanly.
    """

    for attempt in range(max_retries + 1):

        try:
            return llm.invoke(prompt)

        except Exception as error:

            if is_request_too_large(error) and attempt < max_retries:
                requested, limit = _requested_and_limit(error)
                current = getattr(llm, "max_tokens", None) or 4096
                budget = current - (requested - limit) - 150 if requested and limit else current // 2
                if budget >= 1200:
                    print(f"   ↘️ Request too large; max_tokens {current} -> {budget}")
                    llm = llm.model_copy(update={"max_tokens": int(budget)})
                    continue

            error_text = str(error).lower()

            is_daily_quota = is_daily_quota_error(error)

            if is_daily_quota:
                print(
                    "   🛑 [Paper Analysis Agent]: "
                    f"Groq daily token quota is exhausted: {str(error)[:300]}"
                )
                print(
                    "   ↪ Halting the literature-review generation "
                    "without further LLM retries."
                )
                raise GroqQuotaExceeded(
                    "Groq daily token quota exhausted. "
                    "Generation halted because the required token quota "
                    "is not available."
                ) from error

            is_rate_limit = (
                "429" in error_text
                or "rate limit" in error_text
                or "rate_limit_exceeded" in error_text
                or "tokens per minute" in error_text
                or "tpm" in error_text
            )

            if not is_rate_limit:
                raise

            if attempt >= max_retries:
                if strict_mode():
                    # Benchmark runs must not silently skip evidence chunks.
                    raise GroqQuotaExceeded(
                        f"Rate limit persisted after {max_retries} retries: {error}"
                    ) from error
                raise

            wait_time = min(
                90.0,
                max(base_wait * (2 ** attempt), retry_after_seconds(error, 0) + 2),
            )

            print(
                f"   ⏳ Temporary Groq rate limit reached. "
                f"Waiting {wait_time:.0f}s before retry "
                f"({attempt + 1}/{max_retries})..."
            )

            time.sleep(wait_time)


# ============================================================
# TEXT CLEANING
# ============================================================

# Cleans extracted PDF text but keeps the academic structure: unifies line endings,
# squeezes repeated spaces, removes odd characters and extra blank lines.
def clean_paper_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving academic structure.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# CHUNKING
# ============================================================

# Cuts long paper text into pieces the LLM can read (about 14,000 characters each).
#  - Short text stays as one chunk.
#  - Otherwise paragraphs are added to the current chunk until the limit is reached.
#  - A new chunk starts with the last ~1,200 characters of the previous chunk (overlap) so no sentence
#    or idea is lost at the boundary.
def split_into_chunks(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS
) -> List[str]:
    """
    Split long academic text into overlapping chunks.

    Tries to split at paragraph boundaries instead of
    cutting sentences randomly.
    """

    text = clean_paper_text(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paragraphs = text.split("\n\n")

    chunks = []
    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        candidate = (
            current + "\n\n" + paragraph
            if current
            else paragraph
        )

        if len(candidate) <= max_chars:

            current = candidate

        else:

            if current:
                chunks.append(current)

            # Keep a small overlap from previous chunk.
            if overlap > 0 and current:

                overlap_text = current[-overlap:]

                # Avoid starting in the middle of a word.
                first_space = overlap_text.find(" ")

                if first_space >= 0:

                    overlap_text = (
                        overlap_text[first_space + 1:]
                    )

                current = (
                    overlap_text
                    + "\n\n"
                    + paragraph
                )

            else:

                current = paragraph

    if current:
        chunks.append(current)

    return chunks


# ============================================================
# JSON EXTRACTION
# ============================================================

# Pulls the JSON object out of an LLM answer (removes markdown fences, finds the first {...} block).
def extract_json(text: str) -> Dict[str, Any]:
    """
    Safely extract a JSON object from an LLM response.
    """

    text = str(text).strip()

    # Remove markdown fences
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    # Direct JSON
    try:

        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:

        raise ValueError(
            "No JSON object found in LLM response."
        )

    return json.loads(
        text[start:end + 1]
    )


# ============================================================
# NORMALIZATION
# ============================================================

# Makes sure a value is a clean list of non-empty strings: None -> [], a list is cleaned item by item,
# any single value becomes a one-item list.
def ensure_list(value) -> List[str]:

    if value is None:
        return []

    if isinstance(value, list):

        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    return [str(value).strip()]


# Fits one paper analysis into the fixed 'evidence schema' used everywhere else in the project.
# It makes sure all fields exist (study type, research problem, objectives, methodology, dataset, models,
# metrics, key findings, contributions, limitations, future work, evidence records, coverage info),
# cleans every list and keeps evidence records in a consistent format.
def _normalize_analysis_core(
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Normalize one paper analysis into the V2 evidence-grounded schema.
    """

    if not isinstance(analysis, dict):
        analysis = {}

    # --------------------------------------------------------
    # Scalar fields
    # --------------------------------------------------------

    scalar_fields = [
        "study_type",
        "research_problem"
    ]

    for field in scalar_fields:

        value = analysis.get(
            field,
            ""
        )

        if value is None:
            value = ""

        analysis[field] = str(
            value
        ).strip()

    # --------------------------------------------------------
    # List fields
    # --------------------------------------------------------

    list_fields = [
        "objectives",
        "methodology",
        "dataset",
        "models_and_techniques",
        "evaluation_metrics",
        "key_findings",
        "contributions",
        "limitations",
        "future_work"
    ]

    for field in list_fields:

        analysis[field] = ensure_list(
            analysis.get(
                field,
                []
            )
        )

    # --------------------------------------------------------
    # Structured evidence records
    # --------------------------------------------------------

    evidence_records = analysis.get(
        "evidence_records",
        []
    )

    if not isinstance(
        evidence_records,
        list
    ):
        evidence_records = []

    normalized_records = []

    for record in evidence_records:

        if not isinstance(
            record,
            dict
        ):
            continue

        normalized_record = {

            "claim": str(
                record.get(
                    "claim",
                    ""
                )
            ).strip(),

            "evidence": str(
                record.get(
                    "evidence",
                    ""
                )
            ).strip(),

            "section": str(
                record.get(
                    "section",
                    ""
                )
            ).strip(),

            "page": record.get(
                "page",
                None
            ),

            "chunk": record.get(
                "chunk",
                None
            ),

            "evidence_type": str(
                record.get(
                    "evidence_type",
                    ""
                )
            ).strip(),

            "status": str(
                record.get(
                    "status",
                    ""
                )
            ).strip()
        }

        # Only retain useful evidence records.
        if (
            normalized_record["claim"]
            or normalized_record["evidence"]
        ):

            normalized_records.append(
                normalized_record
            )

    analysis[
        "evidence_records"
    ] = normalized_records

    # --------------------------------------------------------
    # Coverage information
    # --------------------------------------------------------

    coverage = analysis.get(
        "coverage",
        {}
    )

    if not isinstance(
        coverage,
        dict
    ):
        coverage = {}

    coverage_defaults = {

        "text_available": False,

        "chunks_available": 0,

        "chunks_analyzed": 0,

        "has_methodology_evidence": False,

        "has_results_evidence": False,

        "has_limitation_evidence": False,

        "has_future_work_evidence": False
    }

    for key, default_value in coverage_defaults.items():

        if key not in coverage:

            coverage[key] = default_value

    analysis[
        "coverage"
    ] = coverage

    # --------------------------------------------------------
    # Safe fallback
    # --------------------------------------------------------

    if not analysis[
        "research_problem"
    ]:

        analysis[
            "research_problem"
        ] = (
            "Not clearly reported in the analyzed text."
        )

    if not analysis[
        "study_type"
    ]:

        analysis[
            "study_type"
        ] = (
            "Not clearly classified from the analyzed text."
        )

    return analysis


# ============================================================
# CHUNK ANALYSIS
# ============================================================

EVIDENCE_TYPE_TO_FIELD = {
    "problem": "research_problem",
    "objective": "objectives",
    "methodology": "methodology",
    "method": "methodology",
    "dataset": "dataset",
    "data": "dataset",
    "technique": "models_and_techniques",
    "model": "models_and_techniques",
    "metric": "evaluation_metrics",
    "evaluation_metric": "evaluation_metrics",
    "reported_result": "key_findings",
    "result": "key_findings",
    "finding": "key_findings",
    "comparison": "key_findings",
    "contribution": "contributions",
    "limitation": "limitations",
    "future_work": "future_work",
}


# Cleans an analysis and fills empty summary fields from the typed evidence records.
# Models sometimes fill only evidence_records; this function copies each record's claim into the matching
# field (for example a 'limitation' record into limitations). Nothing is added that the model did not extract.
def normalize_analysis(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise an analysis and back-fill empty summary fields from the model's
    own typed evidence_records (with low reasoning effort GPT-OSS often fills
    only evidence_records). Nothing is added that the model did not extract.
    """
    analysis = _normalize_analysis_core(result)
    records = analysis.get("evidence_records") or []
    for record in records:
        if not isinstance(record, dict):
            continue
        field = EVIDENCE_TYPE_TO_FIELD.get(str(record.get("evidence_type", "")).strip().lower())
        claim = str(record.get("claim", "")).strip()
        if not field or not claim:
            continue
        if field == "research_problem":
            if not str(analysis.get("research_problem", "")).strip():
                analysis["research_problem"] = claim
            continue
        values = analysis.setdefault(field, [])
        if isinstance(values, list) and claim not in values:
            values.append(claim)
    return analysis


# Extracts study information from ONE chunk of a paper using the LLM.
# The prompt gives 17 strict rules: use only the supplied text, never invent datasets/results/limitations,
# mark things as 'reported', 'proposed' or 'conceptual', and return only JSON.
# The answer has the fields: study type, problem, objectives, methodology, dataset, models, metrics, findings,
# contributions, limitations, future work and evidence records (claim + supporting statement + section).
def analyze_chunk(
    llm: ChatGroq,
    paper: Dict[str, Any],
    chunk: str,
    chunk_number: int,
    total_chunks: int,
    section_hint: str = "Unknown"
) -> Dict[str, Any]:
    """
    Extract evidence-grounded study information from one paper chunk.

    Every important claim should be traceable to the supplied text.
    """

    title = paper.get(
        "title",
        "Unknown paper"
    )

    prompt = f"""
You are an academic literature-review evidence extraction agent.

Your task is to extract structured, evidence-grounded information
from ONLY the supplied text of a research paper.

Paper title:
{title}

Chunk:
{chunk_number} of {total_chunks}

Likely section:
{section_hint}

CRITICAL RULES:

1. Use ONLY information explicitly supported by the supplied text.
2. Do NOT use outside knowledge.
3. Do NOT invent datasets, methods, results, metrics, limitations,
   contributions, or research gaps.
4. Do NOT infer numerical results that are not explicitly stated.
5. Distinguish what the paper REPORTS from what the paper PROPOSES.
6. If a method is proposed by the authors, mark its status as
   "proposed".
7. If a result is experimentally reported, mark its status as
   "reported".
8. If something is conceptual rather than experimentally demonstrated,
   mark it as "conceptual".
9. Preserve the terminology used by the paper.
10. Evidence must be traceable to the supplied text.
11. Do not create citations.
12. Do not evaluate the paper using outside knowledge.
13. Do not turn an author's claim into an independently verified fact.
14. If information is absent, return an empty list.
15. Return ONLY valid JSON.
16. Do not use Markdown.
17. Do not add explanations before or after the JSON.

For evidence_records:

- claim = concise statement of what the paper says.
- evidence = concise supporting statement from the supplied text.
- section = section name if identifiable; otherwise use the supplied
  likely section.
- page = use null because page information is not guaranteed here.
- chunk = {chunk_number}
- evidence_type should describe the role of the evidence.

Allowed evidence_type examples:

"problem"
"objective"
"methodology"
"dataset"
"technique"
"reported_result"
"contribution"
"limitation"
"future_work"
"comparison"
"other"

Allowed status values:

"reported"
"proposed"
"conceptual"
"observed"
"future_work"
"not_clear"

Use exactly this JSON structure:

{{
  "study_type": "",
  "research_problem": "",
  "objectives": [],
  "methodology": [],
  "dataset": [],
  "models_and_techniques": [],
  "evaluation_metrics": [],
  "key_findings": [],
  "contributions": [],
  "limitations": [],
  "future_work": [],

  "evidence_records": [
    {{
      "claim": "",
      "evidence": "",
      "section": "",
      "page": null,
      "chunk": {chunk_number},
      "evidence_type": "",
      "status": ""
    }}
  ]
}}

PAPER TEXT:

{chunk}
"""

    # --------------------------------------------------------
    # First attempt
    # --------------------------------------------------------

    response = invoke_with_rate_limit_retry(
        llm,
        prompt
    )

    content = (
        response.content
        if hasattr(
            response,
            "content"
        )
        else str(response)
    )

    try:

        result = extract_json(
            content
        )

        return normalize_analysis(
            result
        )

    except Exception as first_error:

        print(
            f"   ⚠️ JSON parsing failed for chunk "
            f"{chunk_number}: {first_error}"
        )

        # ----------------------------------------------------
        # Strict JSON repair
        # ----------------------------------------------------

        retry_prompt = f"""
Convert the following response into ONE valid JSON object.

Return ONLY JSON.

Do NOT:
- add facts
- invent evidence
- infer results
- remove factual information
- add explanations
- use Markdown

Preserve the supplied information.

Required schema:

{{
  "study_type": "",
  "research_problem": "",
  "objectives": [],
  "methodology": [],
  "dataset": [],
  "models_and_techniques": [],
  "evaluation_metrics": [],
  "key_findings": [],
  "contributions": [],
  "limitations": [],
  "future_work": [],

  "evidence_records": [
    {{
      "claim": "",
      "evidence": "",
      "section": "",
      "page": null,
      "chunk": {chunk_number},
      "evidence_type": "",
      "status": ""
    }}
  ]
}}

LLM RESPONSE:

{content}
"""

        try:

            print(
                f"   🔄 Retrying chunk "
                f"{chunk_number} with strict JSON format..."
            )

            retry_response = invoke_with_rate_limit_retry(
                llm,
                retry_prompt,
                max_retries=1,
                base_wait=5.0
            )

            retry_content = (
                retry_response.content
                if hasattr(
                    retry_response,
                    "content"
                )
                else str(retry_response)
            )

            result = extract_json(
                retry_content
            )

            print(
                f"   ✅ JSON recovered for chunk "
                f"{chunk_number}."
            )

            return normalize_analysis(
                result
            )

        except Exception as retry_error:

            print(
                f"   ❌ JSON recovery failed for chunk "
                f"{chunk_number}: {retry_error}"
            )

            return normalize_analysis({})


# ============================================================
# MERGE ANALYSES
# ============================================================

# Combines the results of all chunks of one paper into a single analysis.
# Scalar fields (study type, research problem) take the first useful value; lists are joined without duplicates;
# evidence records are kept with where they came from. It also records coverage:
# how many chunks were analysed and whether methodology/results/limitation/future-work evidence was found.
def merge_analyses(
    analyses: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge chunk-level analyses while preserving evidence provenance.
    """

    if not analyses:

        return normalize_analysis({})

    merged = {

        "study_type": "",

        "research_problem": "",

        "objectives": [],

        "methodology": [],

        "dataset": [],

        "models_and_techniques": [],

        "evaluation_metrics": [],

        "key_findings": [],

        "contributions": [],

        "limitations": [],

        "future_work": [],

        "evidence_records": [],

        "coverage": {

            "text_available": True,

            "chunks_available": len(
                analyses
            ),

            "chunks_analyzed": len(
                analyses
            ),

            "has_methodology_evidence": False,

            "has_results_evidence": False,

            "has_limitation_evidence": False,

            "has_future_work_evidence": False
        }
    }

    # --------------------------------------------------------
    # Scalar fields
    # --------------------------------------------------------

    for analysis in analyses:

        if not isinstance(
            analysis,
            dict
        ):
            continue

        study_type = analysis.get(
            "study_type",
            ""
        )

        if (
            study_type
            and "not clearly" not in study_type.lower()
            and not merged["study_type"]
        ):

            merged[
                "study_type"
            ] = study_type

        problem = analysis.get(
            "research_problem",
            ""
        )

        if (
            problem
            and "not clearly reported"
            not in problem.lower()
            and not merged["research_problem"]
        ):

            merged[
                "research_problem"
            ] = problem

    # --------------------------------------------------------
    # List fields
    # --------------------------------------------------------

    list_fields = [

        "objectives",

        "methodology",

        "dataset",

        "models_and_techniques",

        "evaluation_metrics",

        "key_findings",

        "contributions",

        "limitations",

        "future_work"
    ]

    for field in list_fields:

        seen = set()

        for analysis in analyses:

            if not isinstance(
                analysis,
                dict
            ):
                continue

            values = analysis.get(
                field,
                []
            )

            if not isinstance(
                values,
                list
            ):
                values = ensure_list(
                    values
                )

            for item in values:

                item = str(
                    item
                ).strip()

                if not item:
                    continue

                normalized = re.sub(
                    r"\s+",
                    " ",
                    item.lower()
                )

                if normalized not in seen:

                    seen.add(
                        normalized
                    )

                    merged[
                        field
                    ].append(
                        item
                    )

    # --------------------------------------------------------
    # Evidence records
    # --------------------------------------------------------

    evidence_seen = set()

    for analysis in analyses:

        if not isinstance(
            analysis,
            dict
        ):
            continue

        records = analysis.get(
            "evidence_records",
            []
        )

        if not isinstance(
            records,
            list
        ):
            continue

        for record in records:

            if not isinstance(
                record,
                dict
            ):
                continue

            normalized_record = {

                "claim": str(
                    record.get(
                        "claim",
                        ""
                    )
                ).strip(),

                "evidence": str(
                    record.get(
                        "evidence",
                        ""
                    )
                ).strip(),

                "section": str(
                    record.get(
                        "section",
                        ""
                    )
                ).strip(),

                "page": record.get(
                    "page",
                    None
                ),

                "chunk": record.get(
                    "chunk",
                    None
                ),

                "evidence_type": str(
                    record.get(
                        "evidence_type",
                        ""
                    )
                ).strip(),

                "status": str(
                    record.get(
                        "status",
                        ""
                    )
                ).strip()
            }

            key = (
                normalized_record[
                    "claim"
                ].lower(),

                normalized_record[
                    "evidence"
                ].lower()
            )

            if (
                not normalized_record["claim"]
                and not normalized_record["evidence"]
            ):
                continue

            if key not in evidence_seen:

                evidence_seen.add(
                    key
                )

                merged[
                    "evidence_records"
                ].append(
                    normalized_record
                )

    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    for analysis in analyses:

        if not isinstance(
            analysis,
            dict
        ):
            continue

        for record in analysis.get(
            "evidence_records",
            []
        ):

            if not isinstance(
                record,
                dict
            ):
                continue

            evidence_type = str(
                record.get(
                    "evidence_type",
                    ""
                )
            ).lower()

            if evidence_type == "methodology":

                merged[
                    "coverage"
                ][
                    "has_methodology_evidence"
                ] = True

            if evidence_type == "reported_result":

                merged[
                    "coverage"
                ][
                    "has_results_evidence"
                ] = True

            if evidence_type == "limitation":

                merged[
                    "coverage"
                ][
                    "has_limitation_evidence"
                ] = True

            if evidence_type == "future_work":

                merged[
                    "coverage"
                ][
                    "has_future_work_evidence"
                ] = True

    return normalize_analysis(
        merged
    )


# ============================================================
# OPTIONAL FINAL REFINEMENT
# ============================================================

# Optional second LLM pass that cleans and consolidates a merged analysis (removes duplicates,
# keeps technical terms, invents nothing). NOTE: analyze_paper() currently does NOT call it -
# the merged chunk evidence is used directly to save tokens and avoid hallucination.
def refine_analysis(
    llm: ChatGroq,
    paper: Dict[str, Any],
    merged_analysis: Dict[str, Any]
) -> Dict[str, Any]:

    title = paper.get(
        "title",
        "Unknown paper"
    )

    prompt = f"""
You are a senior academic literature-review analyst.

You are given structured evidence extracted from a research
paper.

Paper:
{title}

Your task is to clean and consolidate the extracted evidence.

IMPORTANT:

- Use ONLY the supplied evidence.
- Do NOT add outside knowledge.
- Do NOT invent results.
- Do NOT invent datasets.
- Do NOT invent limitations.
- Remove duplicate statements.
- Preserve important technical terminology.
- Keep the output academically precise.
- If a field has no supported evidence, return an empty list.
- Return ONLY valid JSON.

Return exactly:

{{
  "research_problem": "",
  "objectives": [],
  "methodology": [],
  "dataset": [],
  "models_and_techniques": [],
  "key_findings": [],
  "contributions": [],
  "limitations": [],
  "future_work": [],
  "evidence": []
}}

Extracted evidence:

{json.dumps(
    merged_analysis,
    ensure_ascii=False,
    indent=2
)}
"""

    response = llm.invoke(
        prompt
    )

    content = (
        response.content
        if hasattr(
            response,
            "content"
        )
        else str(response)
    )

    result = extract_json(
        content
    )

    return normalize_analysis(
        result
    )


# ============================================================
# MAIN SINGLE-PAPER ANALYSIS AGENT
# ============================================================

# Analyses ONE paper end to end.
#  1. Takes the full text (or the abstract if no full text) and splits it into chunks.
#  2. If there are more than 5 chunks, scores each by section words (introduction, method, results, limitations,
#     conclusion ...) and position, and keeps the first, the last and the best-scoring ones (max 5).
#  3. For every chosen chunk calls analyze_chunk() with gpt-oss-20b (temperature 0).
#  4. merge_analyses() joins the chunk results; coverage information is corrected.
#  5. normalize_analysis() gives the final analysis, which is stored as paper['paper_analysis'].
# A failed chunk is skipped; a quota error stops everything.
def analyze_paper(
    paper: Dict[str, Any]
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not isinstance(
        paper,
        dict
    ):

        print(
            "⚠️ [Paper Analysis Agent]: "
            "Invalid paper record."
        )

        return {
            "paper_analysis":
                normalize_analysis({})
        }

    title = paper.get(
        "title",
        "Unknown paper"
    )

    full_text = paper.get(
        "full_text",
        ""
    )

    # --------------------------------------------------------
    # Fallback to abstract if full text is unavailable
    # --------------------------------------------------------

    if not full_text:

        full_text = paper.get(
            "abstract",
            ""
        )

    if not full_text:

        return {
            **paper,
            "paper_analysis":
                normalize_analysis({})
        }

    print(
        f"\n🧠 [Paper Analysis Agent]: "
        f"Analyzing '{title[:80]}'"
    )

    # --------------------------------------------------------
    # Split paper into chunks
    # --------------------------------------------------------

    chunks = split_into_chunks(
        full_text
    )

    if not chunks:

        chunks = [
            full_text
        ]

    # --------------------------------------------------------
    # Select representative evidence chunks
    # --------------------------------------------------------

    if len(chunks) > MAX_EVIDENCE_CHUNKS:

        print(
            f"⚠️ Paper contains "
            f"{len(chunks)} chunks."
        )

        evidence_keywords = [

            "abstract",

            "introduction",

            "background",

            "problem",

            "objective",

            "motivation",

            "method",

            "methodology",

            "approach",

            "framework",

            "architecture",

            "system",

            "model",

            "algorithm",

            "dataset",

            "data",

            "experiment",

            "evaluation",

            "result",

            "results",

            "finding",

            "findings",

            "discussion",

            "limitation",

            "limitations",

            "future work",

            "conclusion",

            "conclusions",

            "contribution",

            "contributions"
        ]

        high_value_terms = [

            "proposed",

            "we present",

            "we introduce",

            "we develop",

            "we evaluate",

            "we investigate",

            "experimental",

            "performance",

            "accuracy",

            "comparison",

            "observed",

            "findings"
        ]

        scored_chunks = []

        for index, chunk in enumerate(
            chunks
        ):

            chunk_lower = chunk.lower()

            score = 0

            # Academic section indicators
            for keyword in evidence_keywords:

                if keyword in chunk_lower:

                    score += 2

            # Research evidence indicators
            for term in high_value_terms:

                if term in chunk_lower:

                    score += 1

            # Preserve coverage across the paper
            position_score = (
                1.0
                - abs(
                    (
                        index
                        / max(
                            1,
                            len(chunks) - 1
                        )
                    )
                    - 0.5
                )
            )

            score += position_score

            scored_chunks.append(
                (
                    score,
                    index
                )
            )

        # ----------------------------------------------------
        # Rank chunks
        # ----------------------------------------------------

        ranked_indices = [

            index

            for _, index in sorted(
                scored_chunks,
                key=lambda item: item[0],
                reverse=True
            )
        ]

        selected_indices = []

        # Always preserve beginning
        selected_indices.append(
            0
        )

        # Always preserve ending
        last_index = (
            len(chunks) - 1
        )

        if last_index not in selected_indices:

            selected_indices.append(
                last_index
            )

        # Add strongest evidence chunks
        for index in ranked_indices:

            if index not in selected_indices:

                selected_indices.append(
                    index
                )

            if (
                len(selected_indices)
                >= MAX_EVIDENCE_CHUNKS
            ):

                break

        # Keep original paper order
        selected_indices = sorted(
            set(selected_indices)
        )

        chunks = [

            chunks[index]

            for index in selected_indices
        ]

        print(
            f"↪ Selected "
            f"{len(chunks)} "
            f"section-aware evidence chunks."
        )

    print(
        f"📑 [Paper Analysis Agent]: "
        f"{len(chunks)} chunks to analyze."
    )

    # --------------------------------------------------------
    # Initialize LLM
    # --------------------------------------------------------

    llm = make_llm(MODEL_NAME, temperature=0)

    chunk_analyses = []

    # --------------------------------------------------------
    # Analyze each selected chunk
    # --------------------------------------------------------

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"   🔬 Analyzing chunk "
            f"{index}/{len(chunks)}..."
        )

        try:

            analysis = analyze_chunk(
                llm,
                paper,
                chunk,
                index,
                len(chunks),
                section_hint=(
                    "Academic section containing "
                    "relevant research evidence"
                )
            )

            # Never allow None into the analysis list
            if isinstance(
                analysis,
                dict
            ):

                chunk_analyses.append(
                    analysis
                )

            else:

                print(
                    f"   ⚠️ Chunk {index} returned "
                    f"invalid analysis. Skipping."
                )

        except GroqQuotaExceeded:
            # A daily/quota exhaustion is a run-level failure.
            # Do not continue with another paper or another LLM request.
            raise

        except Exception as error:

            print(
                f"   ⚠️ Chunk {index} failed: "
                f"{error}"
            )

            # Temporary/non-quota failures do not destroy the
            # entire paper analysis.
            continue

    # --------------------------------------------------------
    # Merge chunk-level evidence
    # --------------------------------------------------------

    print(
        "🔗 [Paper Analysis Agent]: "
        "Merging extracted evidence..."
    )

    merged = merge_analyses(
        chunk_analyses
    )

    # --------------------------------------------------------
    # Correct coverage information
    # --------------------------------------------------------

    merged[
        "coverage"
    ][
        "chunks_available"
    ] = len(
        split_into_chunks(
            full_text
        )
    )

    merged[
        "coverage"
    ][
        "chunks_analyzed"
    ] = len(
        chunk_analyses
    )

    merged[
        "coverage"
    ][
        "text_available"
    ] = bool(
        full_text.strip()
    )

    # --------------------------------------------------------
    # Final deterministic analysis
    #
    # No additional LLM refinement.
    # --------------------------------------------------------

    print(
        "✨ [Paper Analysis Agent]: "
        "Using merged chunk evidence without "
        "additional LLM refinement."
    )

    final_analysis = normalize_analysis(
        merged
    )

    print(
        "✅ [Paper Analysis Agent]: "
        "Paper analysis completed."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Always return a dictionary.
    # --------------------------------------------------------

    return {
        **paper,
        "paper_analysis":
            final_analysis
    }


# ============================================================
# MULTI-PAPER ANALYSIS
# ============================================================

# Runs analyze_paper() for every paper in the list, one by one, with progress printing.
# If a paper fails, its record is kept with an empty analysis so the run continues;
# if the Groq quota is exhausted, the error is raised so the whole generation halts.
def analyze_papers(
    papers: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    results = []

    print(
        "\n" + "=" * 100
    )

    print(
        "📚 PAPER ANALYSIS"
    )

    print(
        "=" * 100
    )

    if not isinstance(
        papers,
        list
    ):

        print(
            "❌ [Paper Analysis Agent]: "
            "Expected a list of papers."
        )

        return []

    for index, paper in enumerate(
        papers,
        start=1
    ):

        print(
            f"\n[{index}/{len(papers)}]"
        )

        # ----------------------------------------------------
        # Validate paper record
        # ----------------------------------------------------

        if not isinstance(
            paper,
            dict
        ):

            print(
                f"⚠️ Paper {index} is not a dictionary. "
                f"Skipping."
            )

            continue

        try:

            result = analyze_paper(
                paper
            )

            # ------------------------------------------------
            # Never allow None into results
            # ------------------------------------------------

            if isinstance(
                result,
                dict
            ):

                results.append(
                    result
                )

            else:

                print(
                    "⚠️ Paper analysis returned "
                    "an invalid result. "
                    "Using safe fallback record."
                )

                results.append({

                    **paper,

                    "paper_analysis":
                        normalize_analysis({})
                })

        except GroqQuotaExceeded:
            # Do not convert quota exhaustion into an empty paper result.
            # Propagate it to app.py so the complete generation is halted
            # and reported as RATE_LIMITED.
            raise

        except Exception as error:

            print(
                f"❌ Paper analysis failed: "
                f"{error}"
            )

            # Preserve paper metadata even if
            # analysis fails.
            results.append({

                **paper,

                "paper_analysis":
                    normalize_analysis({})
            })

    print(
        "\n" + "=" * 100
    )

    print(
        f"✅ [Paper Analysis Agent]: "
        f"Processed {len(results)} papers."
    )

    print(
        "=" * 100
    )

    return results