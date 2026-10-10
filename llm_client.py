"""
Shared Groq LLM client for every agent in the pipeline.

* Reads the API key only from the environment / .env (no keys in source).
* Retries transient TPM rate limits with back-off.
* Distinguishes daily-quota exhaustion (not retryable today) from
  per-minute throttling (retryable after a short wait).
* "Strict" mode (MALRG_STRICT_LLM=1) is used by the ablation harness:
  instead of silently switching to a deterministic fallback when the LLM
  is unavailable, agents raise LLMUnavailable so a benchmark run is never
  contaminated by fallback text.
"""

import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq

load_dotenv()


# ----------------------------------------------------------------------
# Token accounting (used by the ablation harness for efficiency metrics)
# ----------------------------------------------------------------------
TOKEN_USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


# Sets the global token counters (calls, input tokens, output tokens) back to zero.
# The ablation runner calls it before each stage to measure the cost of that stage.
def reset_token_usage() -> None:
    for key in TOKEN_USAGE:
        TOKEN_USAGE[key] = 0


# Returns a copy of the current token counters so two moments can be compared.
def snapshot_token_usage() -> dict:
    return dict(TOKEN_USAGE)


# A LangChain hook that watches every LLM call and adds up its tokens into TOKEN_USAGE.
class _UsageCallback(BaseCallbackHandler):
    # Runs automatically after each LLM call: counts one call and adds the input and output tokens
    # reported by Groq to the global counters.
    def on_llm_end(self, response, **kwargs):  # noqa: D401 - LangChain hook
        TOKEN_USAGE["calls"] += 1
        counted = False
        for generations in getattr(response, "generations", []) or []:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                if usage:
                    TOKEN_USAGE["input_tokens"] += int(usage.get("input_tokens", 0))
                    TOKEN_USAGE["output_tokens"] += int(usage.get("output_tokens", 0))
                    counted = True
        if not counted:
            usage = (getattr(response, "llm_output", None) or {}).get("token_usage", {}) or {}
            TOKEN_USAGE["input_tokens"] += int(usage.get("prompt_tokens", 0))
            TOKEN_USAGE["output_tokens"] += int(usage.get("completion_tokens", 0))


_USAGE_CALLBACK = _UsageCallback()


# Error meaning 'the LLM cannot be used right now'. In strict mode agents raise it instead of silently
# using fallback text, so benchmark results are never contaminated.
class LLMUnavailable(RuntimeError):
    """Raised (strict mode only) when the LLM cannot be reached."""


# Error meaning Groq's tokens-per-day or requests-per-day quota is used up; retrying today is useless.
class DailyQuotaExhausted(LLMUnavailable):
    """Groq tokens-per-day / requests-per-day quota is used up."""


# Groq free tier allows ~8K tokens per minute per model, so every agent keeps its
# prompt under this budget (leaving room for the answer).
PROMPT_TOKEN_BUDGET = int(os.environ.get("MALRG_PROMPT_TOKEN_BUDGET", "5200"))


# Cheap token estimate: number of characters divided by 3.4. Used to keep prompts under the token budget.
def estimate_tokens(text: str) -> int:
    """Conservative token estimate for English/JSON text (~3.4 chars per token)."""
    return int(len(text or "") / 3.4) + 1


# Reads GROQ_API_KEY from the environment/.env and raises a clear error if it is missing.
# The key is never written in the source code.
def get_groq_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to the .env file in the project root."
        )
    return key


# True when MALRG_STRICT_LLM=1. Used by the benchmark so failures stop the run instead of using fallbacks.
def strict_mode() -> bool:
    return os.environ.get("MALRG_STRICT_LLM", "0").strip() in {"1", "true", "yes"}


# True if an error message says the DAILY quota is gone (tokens per day, requests per day, quota exceeded ...).
def is_daily_quota_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "tokens per day",
            "token per day",
            "requests per day",
            "(tpd)",
            "(rpd)",
            "daily limit",
            "quota exceeded",
            "quota_exceeded",
            "daily quota",
            "insufficient quota",
        )
    )


# True if an error is a rate limit: HTTP 429, 'rate limit', 'tokens per minute' or a daily quota error.
def is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "rate_limit_exceeded" in text
        or "tokens per minute" in text
        or is_daily_quota_error(error)
    )


# True if Groq says the request is larger than the allowed size (prompt + max output tokens).
def is_request_too_large(error: Exception) -> bool:
    text = str(error).lower()
    return "request too large" in text or "error code: 413" in text


# Reads the two numbers from a 'request too large' error message: how many tokens were requested
# and what the limit is. Used to compute a smaller max_tokens.
def _requested_and_limit(error: Exception):
    text = str(error)
    requested = re.search(r"Requested\s+(\d+)", text)
    limit = re.search(r"Limit\s+(\d+)", text)
    return (int(requested.group(1)) if requested else None, int(limit.group(1)) if limit else None)


# Reads how long Groq asks us to wait from the message ('try again in 7.5s' or '2m3.1s').
# Returns the default if no time is found.
def retry_after_seconds(error: Exception, default: float = 20.0) -> float:
    """Parse 'Please try again in 7.5s' / '2m3.1s' from a Groq error."""
    text = str(error)
    match = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", text)
    if not match or not any(match.groups()):
        return default
    hours = float(match.group(1) or 0)
    minutes = float(match.group(2) or 0)
    seconds = float(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


# Creates a Groq chat model for an agent.
# Sets the API key, model name, temperature, up to 6 automatic retries and the token-usage hook.
# For gpt-oss models it also sets max_tokens (default 4096) and reasoning_effort, because these models
# spend tokens on hidden thinking and could otherwise return an empty answer.
def make_llm(
    model: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
) -> ChatGroq:
    kwargs = dict(
        api_key=get_groq_api_key(),
        model=model,
        temperature=temperature,
        max_retries=int(os.environ.get("MALRG_LLM_MAX_RETRIES", "6")),
        callbacks=[_USAGE_CALLBACK],
    )
    is_reasoning_model = model.startswith("openai/gpt-oss")
    # GPT-OSS models spend completion tokens on hidden reasoning; with the
    # provider's small default budget they can return an EMPTY answer.
    if not max_tokens and is_reasoning_model:
        max_tokens = int(os.environ.get("MALRG_DEFAULT_MAX_TOKENS", "4096"))
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    if is_reasoning_model:
        kwargs["reasoning_effort"] = reasoning_effort or os.environ.get("MALRG_REASONING_EFFORT", "low")
    return ChatGroq(**kwargs)


# Error for an empty answer: the model used its whole token budget on reasoning and returned no text.
class EmptyCompletion(RuntimeError):
    """The model used its whole token budget on reasoning and returned no text."""


# Calls an LLM and returns its text, handling the problems of Groq's free tier.
#  - Empty answer: retry once, then raise.
#  - Request too large: reduce max_tokens so prompt + answer fit, and retry.
#  - Daily quota: wait only if the wait is under 2 minutes, otherwise raise DailyQuotaExhausted.
#  - Per-minute rate limit: wait the time Groq suggests (up to 90 s) and retry (max 5 attempts).
def invoke_text(
    llm: ChatGroq,
    prompt: str,
    max_attempts: int = 5,
    label: str = "LLM",
) -> str:
    """
    Invoke an LLM and return its text, waiting through per-minute limits.
    Daily-quota exhaustion is raised immediately as DailyQuotaExhausted.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
            if not str(content).strip():
                raise EmptyCompletion(
                    f"{label}: empty completion (finish_reason="
                    f"{(getattr(response, 'response_metadata', {}) or {}).get('finish_reason')})"
                )
            return str(content)
        except EmptyCompletion as error:
            last_error = error
            print(f"   ⚠️ {error}; retrying ({attempt}/{max_attempts})...")
            if attempt >= 2:
                raise
            continue
        except Exception as error:  # noqa: BLE001 - provider errors vary
            last_error = error
            if is_request_too_large(error):
                # Groq counts prompt + max_tokens against the per-minute limit:
                # shrink the output budget so the request fits, then retry.
                requested, limit = _requested_and_limit(error)
                current = getattr(llm, "max_tokens", None) or 8000
                new_budget = current - (requested - limit) - 150 if requested and limit else current // 2
                if new_budget >= 1200 and attempt < max_attempts:
                    print(f"   ↘️ [{label}]: request too large, max_tokens {current} -> {new_budget}")
                    llm = llm.model_copy(update={"max_tokens": int(new_budget)})
                    continue
                raise LLMUnavailable(f"{label}: prompt too large for the per-minute token limit: {error}") from error
            if is_daily_quota_error(error):
                wait = retry_after_seconds(error, default=3600)
                # Groq reports TPD exhaustion with a retry-after; a short
                # wait (< 2 min) means the rolling window frees up soon.
                if wait <= 120 and attempt < max_attempts:
                    print(f"   ⏳ [{label}]: daily window busy, waiting {wait:.0f}s...")
                    time.sleep(wait + 1)
                    continue
                raise DailyQuotaExhausted(str(error)) from error
            if is_rate_limit_error(error) and attempt < max_attempts:
                wait = min(90.0, retry_after_seconds(error) + 2)
                print(f"   ⏳ [{label}]: rate limited, retrying in {wait:.0f}s ({attempt}/{max_attempts})...")
                time.sleep(wait)
                continue
            raise
    raise LLMUnavailable(str(last_error))
