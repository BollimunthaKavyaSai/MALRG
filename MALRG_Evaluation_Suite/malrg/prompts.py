"""
Writer prompt variants for the prompt-comparison experiment (P1-P5).

Every variant receives the SAME evidence context (sources, extracted evidence, cross-paper
analysis, verified gaps); only the instructions differ, so differences in the output are
attributable to the prompt alone.
"""

STRUCTURE = """## Introduction
2-3 thematic sections (## <theme name>)
## Comparative Analysis
## Research Gaps and Future Directions
## Conclusion"""

PROMPTS = {
    # P1 - structured rules (the system default, also used by the ablation configs)
    "P1": {
        "name": "P1: Structured rules (default)",
        "style": "rule-based instruction",
        "template": """{context}

Write an academic literature review (900-1300 words) in Markdown.
Structure:
""" + STRUCTURE + """
Rules:
- Support every factual statement with numeric citations like [1] or [2, 3].
- Synthesize across papers (compare, contrast, relate); do not describe papers one by one.
- Report concrete methods, datasets and numbers where available.
- Write in prose paragraphs only (no tables, no bullet lists).
- Do not overstate certainty.
- Cite only the numbered sources above. Do not write a reference list.{preferences}""",
    },
    # P2 - minimal zero-shot
    "P2": {
        "name": "P2: Zero-shot minimal",
        "style": "zero-shot",
        "template": """{context}

Write a literature review on this topic using the sources above. Cite sources with [n].
Do not write a reference list.{preferences}""",
    },
    # P3 - expert persona + reviewer criteria (role prompting)
    "P3": {
        "name": "P3: Expert persona + reviewer criteria",
        "style": "role prompting",
        "template": """You are a senior researcher and journal editor who has published many systematic
reviews. Reviewers of your journal reject reviews that merely summarize papers, contain claims not
backed by the cited work, or propose vague research gaps.

{context}

Write the related-work / literature review section (900-1300 words, Markdown) that would pass
this peer review. Use the structure:
""" + STRUCTURE + """
Critically compare methods and evidence, and cite every claim with [n].
Prose paragraphs only. Do not write a reference list.{preferences}""",
    },
    # P4 - plan-then-write (outline first, chain-of-thought style)
    "P4": {
        "name": "P4: Plan-then-write (outline first)",
        "style": "chain-of-thought planning",
        "template": """{context}

Task: write an academic literature review (900-1300 words) in Markdown.
Step 1 - PLAN: under a line "PLAN:", list the themes, which papers support each theme, the key
comparisons you will make, and the gaps, as a short outline with source numbers.
Step 2 - REVIEW: under a line "REVIEW:", write the review following your plan with the structure:
""" + STRUCTURE + """
Cite every claim with [n], compare across papers, prose paragraphs only, no reference list.{preferences}""",
    },
    # P5 - evidence-anchored, claim-level constraints
    "P5": {
        "name": "P5: Evidence-anchored constraints",
        "style": "grounding constraints",
        "template": """{context}

Write an academic literature review (900-1300 words) in Markdown with the structure:
""" + STRUCTURE + """
Grounding constraints (strict):
- Every sentence that states a fact must paraphrase specific evidence given above for the
  source(s) it cites; keep technical terms and numbers exactly as written in the evidence.
- Never add numbers, datasets or results that are not in the evidence.
- Comparative sentences must cite every paper they compare, e.g. "[1] uses X, whereas [3] uses Y".
- Interpretation is allowed only when explicitly marked as such ("this suggests ...") and must
  rest on cited facts.
- Prose paragraphs only. Do not write a reference list.{preferences}""",
    },
}


def extract_review(text, variant):
    """P4 returns PLAN + REVIEW; keep only the review part."""
    if variant == "P4" and "REVIEW:" in text:
        return text.split("REVIEW:", 1)[1].strip()
    return text
