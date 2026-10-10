import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()
from langchain_groq import ChatGroq


# ============================================================
# FALLBACK QUERY UNDERSTANDING
# ============================================================

# Backup plan when the LLM cannot be used: builds a simple search plan from the topic words.
#  1. Splits the topic into words and removes filler words (the, of, using, based, system ...).
#  2. The first 3 meaningful words become the core concept; all meaningful words become related concepts.
#  3. Search queries are just: the topic, '<topic> research' and '<topic> review'.
# It returns the same fields as the LLM version, so the rest of the pipeline works unchanged.
def fallback_query_understanding(topic: str) -> Dict[str, Any]:
    """
    Generic fallback used when the LLM is unavailable.
    It does not assume any specific research domain.
    """

    topic = topic.strip()

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9-]*\b",
        topic
    )

    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "in",
        "on",
        "of",
        "to",
        "with",
        "using",
        "based",
        "from",
        "into",
        "through",
        "systems",
        "system"
    }

    meaningful_words = [
        word.lower()
        for word in words
        if word.lower() not in stop_words
    ]

    core_concepts = [
        " ".join(meaningful_words[:3])
    ]

    return {
        "original_query": topic,
        "research_intent": topic,
        "core_concepts": core_concepts,
        "domain": [],
        "subtopics": [],
        "related_concepts": meaningful_words,
        "search_queries": [
            topic,
            f"{topic} research",
            f"{topic} review"
        ]
    }


# ============================================================
# JSON EXTRACTION
# ============================================================

# Pulls a JSON object out of the LLM answer even if the LLM wrapped it in markdown.
# Removes ```json fences, tries to parse the whole text, and if that fails takes the part between
# the first '{' and the last '}'. Raises an error if there is no JSON at all.
def extract_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON even if the LLM surrounds it with markdown.
    """

    text = text.strip()

    # Remove markdown code fences.
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

    # Try direct JSON first.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the first JSON object.
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if not match:
        raise ValueError(
            "No valid JSON object found in LLM response."
        )

    return json.loads(
        match.group(0)
    )


# ============================================================
# VALIDATE RESULT
# ============================================================

# Checks and cleans the LLM's query-understanding answer so later code can trust it.
#  1. Adds any missing field (research_intent, core_concepts, domain, subtopics, related_concepts, search_queries).
#  2. Makes sure list fields are lists of non-empty strings.
#  3. Limits the sizes: 5 concepts, 5 domains, 8 subtopics, 12 related concepts, 8 queries.
#  4. Guarantees that the user's original topic is the first search query.
def validate_intent(
    data: Dict[str, Any],
    topic: str
) -> Dict[str, Any]:

    required_fields = [
        "research_intent",
        "core_concepts",
        "domain",
        "subtopics",
        "related_concepts",
        "search_queries"
    ]

    for field in required_fields:

        if field not in data:
            data[field] = []

    # Ensure list fields are actually lists.
    list_fields = [
        "core_concepts",
        "domain",
        "subtopics",
        "related_concepts",
        "search_queries"
    ]

    for field in list_fields:

        if not isinstance(
            data[field],
            list
        ):
            data[field] = [str(data[field])]

        data[field] = [
            str(item).strip()
            for item in data[field]
            if str(item).strip()
        ]

    data["original_query"] = topic

    # --------------------------------------------------------
    # Safety limits
    # --------------------------------------------------------

    data["core_concepts"] = (
        data["core_concepts"][:5]
    )

    data["domain"] = (
        data["domain"][:5]
    )

    data["subtopics"] = (
        data["subtopics"][:8]
    )

    data["related_concepts"] = (
        data["related_concepts"][:12]
    )

    data["search_queries"] = (
        data["search_queries"][:8]
    )

    # --------------------------------------------------------
    # Guarantee at least the original query.
    # --------------------------------------------------------

    if not data["search_queries"]:
        data["search_queries"] = [
            topic
        ]

    if topic not in data["search_queries"]:
        data["search_queries"].insert(
            0,
            topic
        )

    # Maximum 8 queries.
    data["search_queries"] = (
        data["search_queries"][:8]
    )

    return data


# ============================================================
# MAIN QUERY UNDERSTANDING AGENT
# ============================================================

# Query Understanding agent: turns the user's topic into a search plan.
#  1. Sends the topic to gpt-oss-20b (temperature 0) with strict rules: be domain-agnostic, expand with
#     synonyms, give 4-8 scholarly search queries, never invent papers.
#  2. extract_json() reads the JSON answer and validate_intent() cleans it.
#  3. If anything fails, fallback_query_understanding() is used instead.
# The result (intent, core concepts, subtopics, related concepts, queries) is used by the Search agent
# for MongoDB search, ranking and the topic relevance gate.
def understand_research_query(
    topic: str
) -> Dict[str, Any]:

    topic = topic.strip()

    if not topic:
        raise ValueError(
            "Research topic cannot be empty."
        )

    print(
        f"\n🧠 [Query Understanding Agent]: "
        f"Analyzing '{topic}'"
    )

    prompt = f"""
You are a scholarly research-query understanding agent.

Analyze the following research topic:

"{topic}"

Your task is NOT to find papers.
Your task is NOT to invent papers.
Your task is ONLY to understand the research intent and
generate high-quality scholarly search formulations.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "research_intent": "one concise description of what the user wants to study",
  "core_concepts": [
    "main research concept 1",
    "main research concept 2"
  ],
  "domain": [
    "application domain 1"
  ],
  "subtopics": [
    "important subtopic 1",
    "important subtopic 2"
  ],
  "related_concepts": [
    "synonym or closely related research concept"
  ],
  "search_queries": [
    "targeted scholarly search query 1",
    "targeted scholarly search query 2",
    "targeted scholarly search query 3",
    "targeted scholarly search query 4",
    "targeted scholarly search query 5"
  ]
}}

Rules:

1. Be domain-agnostic.
2. Never assume the domain is healthcare.
3. Identify the actual application domain from the query.
4. Preserve the user's core research concept.
5. Expand terminology using academically meaningful synonyms.
6. Search queries must be useful for scholarly databases.
7. Do not invent paper titles, authors, datasets, results, or citations.
8. Keep search queries concise.
9. Produce 4 to 8 search queries.
10. Do not add explanations outside the JSON.
"""

    try:

        from llm_client import make_llm

        llm = make_llm("openai/gpt-oss-20b", temperature=0)

        response = llm.invoke(prompt)

        content = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        data = extract_json(
            content
        )

        data = validate_intent(
            data,
            topic
        )

        print(
            "✅ [Query Understanding Agent]: "
            "Research intent identified."
        )

        return data

    except Exception as error:

        print(
            "⚠️ [Query Understanding Agent]: "
            f"LLM analysis failed: {error}"
        )

        print(
            "↪ Using deterministic fallback."
        )

        return fallback_query_understanding(
            topic
        )