"""
User-feedback memory: feedback given on one review is distilled into persistent preference
rules that are injected into every FUTURE generation ("act accordingly next time").

  record_feedback(text)      -> stores the raw feedback, updates the distilled rule set
  preferences_block()        -> the prompt block the writer receives
  check_compliance(md, rules)-> automatic compliance check for checkable rules

Two stores are kept apart:
  data/user_preferences.json        the real user's preferences (CLI: run_all.py feedback "...")
  data/experiment_preferences.json  the simulated-feedback experiment (never mixed with the real store)
"""

import json
import re
import time

import config
from malrg import llm
from malrg.text_utils import body_sentences, parse_citations, split_markdown_sections

USER_STORE = config.DATA_DIR / "user_preferences.json"
EXPERIMENT_STORE = config.DATA_DIR / "experiment_preferences.json"


def load(store=USER_STORE):
    if store.exists():
        return json.loads(store.read_text(encoding="utf-8"))
    return {"feedback": [], "rules": []}


def save(state, store=USER_STORE):
    store.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")


def distill(existing_rules, feedback_text):
    """Merge new feedback into the rule set (newer feedback wins on conflicts)."""
    current = "\n".join(f"- {r['rule']}" for r in existing_rules) or "(none)"
    prompt = f"""You maintain the persistent writing preferences of a user of a literature-review generator.
Merge the NEW FEEDBACK into the CURRENT PREFERENCES. Rewrite feedback as short, general, imperative
rules that apply to every future review (not to one topic). If new feedback contradicts an existing
rule, keep only the newer one. Drop duplicates.

CURRENT PREFERENCES:
{current}

NEW FEEDBACK:
"{feedback_text}"

Return JSON {{"rules": [{{"rule": "imperative instruction", "category": "length|structure|content|style|citation"}}]}}"""
    out = llm.chat_json([{"role": "user", "content": prompt}], role="query", max_tokens=700, default={})
    rules = [r for r in (out or {}).get("rules", []) if isinstance(r, dict) and r.get("rule")]
    return rules or existing_rules


def record_feedback(text, topic=None, store=USER_STORE):
    state = load(store)
    state["feedback"].append({"text": text, "topic": topic, "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    state["rules"] = distill(state["rules"], text)
    save(state, store)
    return state["rules"]


def preferences_block(store=USER_STORE):
    rules = load(store)["rules"]
    if not rules:
        return ""
    lines = "\n".join(f"- {r['rule']}" for r in rules)
    return ("\n\nUSER PREFERENCES (learned from the user's earlier feedback; they override the "
            f"default rules above when they conflict):\n{lines}")


# ---------------------------------------------------------------------------
# Simulated feedback for the automatic "next time" experiment
# Each item: what a user might say after reading a review + an automatic check.
# ---------------------------------------------------------------------------
def _body_words(md):
    return sum(len(s.split()) for _, s in body_sentences(md))


def _has_heading(md, pattern):
    return any(re.search(pattern, h, re.I) for h, _ in split_markdown_sections(md))


SIMULATED_FEEDBACK = [
    {"id": "F1", "text": "The review is too long for my report. Keep it concise, under 900 words.",
     "check": lambda md: _body_words(md) <= 900, "label": "Length <= 900 words"},
    {"id": "F2", "text": "Please always add a dedicated section called 'Datasets and Evaluation Metrics' "
                         "that summarizes the datasets and metrics used by the papers.",
     "check": lambda md: _has_heading(md, r"dataset"), "label": "Datasets & metrics section"},
    {"id": "F3", "text": "I want concrete quantitative results. Report the actual numbers (accuracy, F1, "
                         "error rates, etc.) from the papers wherever they are available.",
     "check": lambda md: sum(1 for _, s in body_sentences(md)
                             if re.search(r"\d+(\.\d+)?\s*%|\b0\.\d+\b|\b\d+\.\d+\b", s)
                             and parse_citations(s)) >= 6,
     "label": ">= 6 cited sentences with numbers"},
    {"id": "F4", "text": "Do not use first-person language such as 'we', 'our' or 'us'.",
     "check": lambda md: not any(re.search(r"\b(we|our|us)\b", s, re.I) for _, s in body_sentences(md)),
     "label": "No first-person language"},
    {"id": "F5", "text": "End every review with a short 'Key Takeaways' section just before the conclusion.",
     "check": lambda md: _has_heading(md, r"take-?aways?"), "label": "Key Takeaways section"},
]


def check_compliance(md):
    res = {f["id"]: bool(f["check"](md)) for f in SIMULATED_FEEDBACK}
    res["compliance_rate"] = sum(res.values()) / len(SIMULATED_FEEDBACK)
    return res


def ensure_experiment_preferences():
    """Record the simulated feedback once (as if given after the first topic's review)."""
    state = load(EXPERIMENT_STORE)
    if state["rules"]:
        return state["rules"]
    for f in SIMULATED_FEEDBACK:
        record_feedback(f["text"], topic=config.TOPICS[0], store=EXPERIMENT_STORE)
    return load(EXPERIMENT_STORE)["rules"]
