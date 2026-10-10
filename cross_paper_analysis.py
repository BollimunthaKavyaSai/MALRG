import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()
from langchain_groq import ChatGroq
from llm_client import PROMPT_TOKEN_BUDGET, LLMUnavailable, estimate_tokens, invoke_text, make_llm, strict_mode


MODEL_NAME = "openai/gpt-oss-120b"

# Keep the cross-paper prompt deliberately compact.
# This avoids the TPM problem we encountered in Section 2.2.
MAX_ITEMS_PER_FIELD = 8


# ============================================================
# HELPERS
# ============================================================

# Turns a value into clean one-line text (squeezes whitespace, None becomes '').
def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip()


# Shortens a list: cleans every item, drops empty ones and keeps only the first few.
# Keeps the LLM prompt small.
def compact_list(
    values: Any,
    limit: int = MAX_ITEMS_PER_FIELD
) -> List[str]:

    if not isinstance(values, list):
        if values:
            return [clean_text(values)]
        return []

    result = []
    seen = set()

    for value in values:

        value = clean_text(value)

        if not value:
            continue

        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

        if len(result) >= limit:
            break

    return result


# Pulls the JSON object out of the LLM answer (removes markdown fences, finds the first {...} block).
def extract_json(text: str) -> Dict[str, Any]:

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

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No JSON object found in LLM response."
        )

    return json.loads(
        text[start:end + 1]
    )


# ============================================================
# PREPARE PAPER EVIDENCE
# ============================================================

# Builds the compact evidence pack for every paper that is sent to the cross-paper LLM.
# For each paper it keeps a limited number of items per field (methodology, findings, limitations ...),
# a limited number of evidence records (claim + short supporting text), and gives it a paper id,
# so the LLM can point to papers by id in its answer.
def prepare_paper_evidence(
    papers: List[Dict[str, Any]],
    items: int = MAX_ITEMS_PER_FIELD,
    max_records: int = 12,
    evidence_chars: int = 280,
) -> List[Dict[str, Any]]:

    prepared = []

    for index, paper in enumerate(papers, start=1):

        analysis = paper.get("paper_analysis", {})

        raw_records = analysis.get(
            "evidence_records",
            analysis.get("raw_records", [])
        )

        evidence_records = []

        if isinstance(raw_records, list):
            for record in raw_records:
                if not isinstance(record, dict):
                    continue

                claim = clean_text(record.get("claim", ""))
                evidence_text = clean_text(record.get("evidence", ""))

                if not claim and not evidence_text:
                    continue

                if len(evidence_records) >= max_records:
                    break

                evidence_records.append({
                    "claim": claim,
                    "evidence": evidence_text[:evidence_chars],
                    "evidence_type": clean_text(
                        record.get("evidence_type", "")
                    ),
                    "status": clean_text(
                        record.get("status", "")
                    )
                })

        prepared.append({

            "paper_id": index,

            "title": clean_text(
                paper.get(
                    "title",
                    f"Paper {index}"
                )
            ),

            "year": paper.get("year"),

            "authors": compact_list(
                paper.get("authors", []),
                5
            ),

            "study_type": clean_text(
                analysis.get(
                    "study_type",
                    ""
                )
            ),

            "research_problem": clean_text(
                analysis.get(
                    "research_problem",
                    ""
                )
            ),

            "objectives": compact_list(
                analysis.get(
                    "objectives",
                    []
                ),
                items
            ),

            "methodology": compact_list(
                analysis.get(
                    "methodology",
                    []
                ),
                items
            ),

            "dataset": compact_list(
                analysis.get(
                    "dataset",
                    []
                ),
                items
            ),

            "models_and_techniques": compact_list(
                analysis.get(
                    "models_and_techniques",
                    []
                ),
                items
            ),

            "evaluation_metrics": compact_list(
                analysis.get(
                    "evaluation_metrics",
                    []
                ),
                items
            ),

            "key_findings": compact_list(
                analysis.get(
                    "key_findings",
                    []
                ),
                items
            ),

            "contributions": compact_list(
                analysis.get(
                    "contributions",
                    []
                ),
                items
            ),

            "limitations": compact_list(
                analysis.get(
                    "limitations",
                    []
                ),
                items
            ),

            "future_work": compact_list(
                analysis.get(
                    "future_work",
                    []
                ),
                items
            ),

            "evidence_records": evidence_records,
        })

    return prepared
# ============================================================
# NORMALIZATION
# ============================================================

# Cleans the cross-paper answer into a fixed layout: overall synthesis, common themes, theme evidence,
# methodology comparison, finding comparison, agreements, disagreements, trends, limitations, gaps.
# Missing fields become empty lists and every list is cleaned.
def normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize cross-paper synthesis while preserving
    evidence links needed for research-paper-quality
    Related Work generation.
    """

    if not isinstance(result, dict):
        result = {}

    # Inside normalize_result: turns a value into a list of cleaned non-empty text items ([] if it is not a list).
    def normalize_list(value):
        if not isinstance(value, list):
            return []

        cleaned = []

        for item in value:
            if isinstance(item, str):
                item = clean_text(item)
                if item:
                    cleaned.append(item)

            elif isinstance(item, dict):
                cleaned.append(item)

        return cleaned[:MAX_ITEMS_PER_FIELD]

    # Inside normalize_result: cleans evidence records (theme/comparison + paper ids + supporting evidence)
    # and keeps only the allowed keys.
    def normalize_evidence_records(value, keys):
        if not isinstance(value, list):
            return []

        cleaned = []

        for item in value:
            if not isinstance(item, dict):
                continue

            record = {}

            for key in keys:
                value_item = item.get(key, "")

                if key in {"paper_ids"}:
                    if isinstance(value_item, list):
                        record[key] = value_item[:MAX_ITEMS_PER_FIELD]
                    else:
                        record[key] = []

                elif key in {"supporting_evidence", "evidence", "basis"}:
                    if isinstance(value_item, list):
                        record[key] = [
                            clean_text(x)
                            for x in value_item
                            if clean_text(x)
                        ][:MAX_ITEMS_PER_FIELD]
                    else:
                        record[key] = []

                else:
                    record[key] = clean_text(value_item)

            if any(
                value_item
                for value_item in record.values()
                if value_item
            ):
                cleaned.append(record)

        return cleaned[:MAX_ITEMS_PER_FIELD]

    normalized = {
        "overall_synthesis": clean_text(
            result.get("overall_synthesis", "")
        ),

        "common_themes": normalize_list(
            result.get("common_themes", [])
        ),

        "theme_evidence": normalize_evidence_records(
            result.get("theme_evidence", []),
            [
                "theme",
                "paper_ids",
                "supporting_evidence"
            ]
        ),

        "methodology_comparison": normalize_list(
            result.get("methodology_comparison", [])
        ),

        "comparison_evidence": normalize_evidence_records(
            result.get("comparison_evidence", []),
            [
                "comparison",
                "paper_ids",
                "evidence"
            ]
        ),

        "finding_comparison": normalize_list(
            result.get("finding_comparison", [])
        ),

        "agreements": normalize_list(
            result.get("agreements", [])
        ),

        "disagreements": normalize_list(
            result.get("disagreements", [])
        ),

        "research_trends": normalize_list(
            result.get("research_trends", [])
        ),

        "cross_paper_limitations": normalize_list(
            result.get("cross_paper_limitations", [])
        ),

        "research_gaps": normalize_list(
            result.get("research_gaps", [])
        ),

        "gap_evidence": normalize_evidence_records(
            result.get("gap_evidence", []),
            [
                "gap",
                "paper_ids",
                "basis"
            ]
        ),

        "future_research_directions": normalize_list(
            result.get("future_research_directions", [])
        )
    }

    return normalized

# Cross-Paper Analysis: compares all analysed papers with each other.
#  1. prepare_paper_evidence() builds a compact evidence pack for every paper.
#  2. A strict prompt for gpt-oss-120b says: use only the supplied input, every claim must point to paper ids,
#     keep 'reported' vs 'proposed' separate, return only JSON in a fixed structure.
#  3. The answer is parsed and normalised: common themes, methodology comparison, finding comparison,
#     agreements, disagreements, trends, cross-paper limitations and candidate gaps.
# The writer uses this to write comparison sentences like '[1] relies on X, whereas [4] uses Y'.
def analyze_cross_papers(
    papers: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not papers:

        return normalize_result({
            "overall_synthesis":
                "No papers were provided.",
        })

    print("\n")
    print("=" * 100)
    print("ðŸ”— CROSS-PAPER ANALYSIS AGENT")
    print("=" * 100)

    print(
        f"\nðŸ“š Papers supplied: {len(papers)}"
    )

    evidence = prepare_paper_evidence(
        papers
    )

    print(
        "ðŸ§© Preparing compact evidence representation..."
    )

    compact_context = json.dumps(
        evidence,
        ensure_ascii=False,
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = make_llm(MODEL_NAME, temperature=0)

    prompt = f"""
You are the cross-paper synthesis agent in a multi-agent
academic literature review generation system.

The final downstream goal is to produce a research-paper-quality
Related Work / Literature Review section.

Your task is NOT to summarize papers independently.

Your task is to synthesize the supplied studies into thematic,
evidence-backed relationships that a later writing agent can
turn into coherent academic paragraphs.

INPUT PAPERS AND EXTRACTED EVIDENCE:
{compact_context}

STRICT EVIDENCE RULES:

1. Use ONLY the information supplied in the input.
2. Do NOT use outside knowledge.
3. Do NOT invent methods, datasets, findings, limitations,
   comparisons, trends, or research gaps.
4. Every important synthesis claim must be traceable to one or
   more supplied paper_ids.
5. Preserve the distinction between reported, proposed,
   conceptual, observed, and future-work statements.
6. Do not treat a proposed method as an experimentally validated
   result.
7. Do not infer quantitative superiority unless the supplied
   evidence explicitly supports it.
8. Do not claim that a paper lacks something merely because the
   field is absent from its extracted evidence.
9. Only report disagreements when the supplied evidence supports
   an actual difference or contradiction.
10. Research gaps must be grounded in reported limitations,
    missing evidence, methodological differences, or explicitly
    stated future work.
11. Do not fabricate citation details.
12. Do not produce paper-by-paper mini summaries.

RELATED-WORK SYNTHESIS RULES:

13. Organize thinking around themes, methods, findings,
    approaches, limitations, and research directions.
14. Prefer relationships involving multiple papers whenever
    the evidence allows it.
15. Compare studies explicitly rather than merely listing them.
16. Identify agreements and differences only when supported.
17. Connect methodological choices to reported findings when
    the evidence supports that connection.
18. Identify research trends only when they emerge from the
    supplied papers.
19. Identify gaps that can logically motivate future research,
    but do not invent a gap merely because something was not
    mentioned.
20. The output will be consumed by a writing agent, so provide
    concise but information-rich synthesis points.

IMPORTANT:

A strong result should enable a writer to produce paragraphs
such as:

- several studies address a common research theme, while
  differing in methodology or scope;
- one group of studies adopts one methodological direction,
  whereas another explores a different approach;
- reported findings show areas of agreement and evidence-based
  differences;
- limitations across studies reveal a defensible research gap;
- future work suggested by the literature motivates further
  investigation.

Do NOT literally write those generic sentences unless the
supplied evidence supports them.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "overall_synthesis": "",

  "common_themes": [],

  "theme_evidence": [
    {{
      "theme": "",
      "paper_ids": [],
      "supporting_evidence": []
    }}
  ],

  "methodology_comparison": [],

  "comparison_evidence": [
    {{
      "comparison": "",
      "paper_ids": [],
      "evidence": []
    }}
  ],

  "finding_comparison": [],

  "agreements": [],

  "disagreements": [],

  "research_trends": [],

  "cross_paper_limitations": [],

  "research_gaps": [],

  "gap_evidence": [
    {{
      "gap": "",
      "paper_ids": [],
      "basis": []
    }}
  ],

  "future_research_directions": []
}}

OUTPUT QUALITY REQUIREMENTS:

- Keep each synthesis item specific.
- Avoid vague statements such as "the papers are important."
- Avoid repeating the same point across multiple fields.
- Use paper_ids to preserve traceability.
- Keep evidence statements faithful to the supplied evidence.
- Do not create unsupported causal claims.
- Do not turn missing information into negative claims.
- Focus on synthesis useful for a research-paper Related Work
  section.
"""

    # Keep the prompt within the per-minute token budget by reducing detail.
    for items, records, chars in ((6, 8, 220), (4, 5, 180), (3, 3, 150), (2, 2, 120), (2, 0, 0)):
        if estimate_tokens(prompt) <= PROMPT_TOKEN_BUDGET:
            break
        smaller = json.dumps(
            prepare_paper_evidence(papers, items, records, chars),
            ensure_ascii=False,
        )
        prompt = prompt.replace(compact_context, smaller)
        compact_context = smaller
    print(f"   📏 Cross-paper prompt ≈ {estimate_tokens(prompt)} tokens")

    try:

        print(
            "ðŸ§  [Cross-Paper Analysis Agent]: "
            "Generating synthesis..."
        )

        content = invoke_text(llm, prompt, label="Cross-Paper Analysis Agent")

        result = extract_json(
            content
        )

        result = normalize_result(
            result
        )

        print(
            "âœ… [Cross-Paper Analysis Agent]: "
            "Synthesis completed."
        )

        return result

    except Exception as error:

        if isinstance(error, LLMUnavailable) or strict_mode():
            # Never hide an unavailable LLM behind an empty synthesis
            # during benchmark runs.
            raise

        print(
            f"âš ï¸ [Cross-Paper Analysis Agent]: "
            f"{error}"
        )

        print(
            "â†ª Returning safe fallback."
        )

        return normalize_result({

            "overall_synthesis":
                "Cross-paper synthesis failed. "
                "The individual paper analyses remain available.",

            "common_themes": [],

            "methodology_comparison": [],

            "finding_comparison": [],

            "agreements": [],

            "disagreements": [],

            "research_trends": [],

            "cross_paper_limitations": [],

            "research_gaps": [],

            "future_research_directions": []
        })


# ============================================================
# DISPLAY
# ============================================================

# Prints the cross-paper result in a readable form in the terminal (overall synthesis, then each section).
def print_cross_paper_analysis(
    result: Dict[str, Any]
) -> None:

    print("\n")
    print("=" * 100)
    print("CROSS-PAPER COMPARATIVE ANALYSIS")
    print("=" * 100)

    print("\nðŸ“Œ Overall Synthesis:")
    print(
        result.get(
            "overall_synthesis",
            ""
        )
    )

    sections = [
        (
            "ðŸ”¹ Common Themes",
            "common_themes"
        ),
        (
            "ðŸ”¬ Methodology Comparison",
            "methodology_comparison"
        ),
        (
            "ðŸ“ˆ Finding Comparison",
            "finding_comparison"
        ),
        (
            "ðŸ¤ Agreements",
            "agreements"
        ),
        (
            "âš–ï¸ Disagreements",
            "disagreements"
        ),
        (
            "ðŸ“Š Research Trends",
            "research_trends"
        ),
        (
            "âš ï¸ Cross-Paper Limitations",
            "cross_paper_limitations"
        ),
        (
            "ðŸ”Ž Research Gaps",
            "research_gaps"
        ),
        (
            "ðŸš€ Future Research Directions",
            "future_research_directions"
        )
    ]

    for heading, field in sections:

        print(f"\n{heading}:")

        values = result.get(
            field,
            []
        )

        if not values:
            print("  â€¢ None identified.")

        else:

            for value in values:
                print(
                    f"  â€¢ {value}"
                )


