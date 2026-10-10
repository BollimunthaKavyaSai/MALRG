"""
Groq LLM client used by every agent and by the evaluation judge.

Features
--------
* Disk cache keyed on (model, messages, params): re-running the study costs
  zero tokens and gives byte-identical outputs (reproducibility).
* Per-model token-bucket throttle that respects Groq's tokens-per-minute cap.
* Retry with server-advised back-off on 429 / 5xx.
* `DailyLimitReached` is raised when a model's daily quota is exhausted so the
  runner can stop cleanly and resume later without losing work.
* Per-call usage accounting (tokens, API latency) for the efficiency analysis.
"""

import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import requests

import config

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class DailyLimitReached(RuntimeError):
    pass


# model -> unix time until which its daily quota is known to be exhausted (skip without HTTP)
EXHAUSTED = {}


def _load_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    env_file = config.PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GROQ_API_KEY not found in environment or Project/.env")


class UsageLog:
    """Collects token/latency usage for the run currently in progress."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = []

    def add(self, role, model, usage, latency, cached):
        self.calls.append({
            "role": role,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_s": round(latency, 3),
            "cached": cached,
        })

    def summary(self):
        return {
            "llm_calls": len(self.calls),
            "total_tokens": sum(c["total_tokens"] for c in self.calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in self.calls),
            "completion_tokens": sum(c["completion_tokens"] for c in self.calls),
            "api_latency_s": round(sum(c["latency_s"] for c in self.calls), 2),
            "by_role": _group(self.calls),
        }


def _group(calls):
    out = defaultdict(lambda: {"calls": 0, "tokens": 0})
    for c in calls:
        out[c["role"]]["calls"] += 1
        out[c["role"]]["tokens"] += c["total_tokens"]
    return dict(out)


USAGE = UsageLog()


class _Throttle:
    def __init__(self, tpm):
        self.tpm = tpm
        self.events = defaultdict(deque)   # model -> deque[(t, tokens)]
        self.lock = threading.Lock()

    def wait(self, model, est_tokens):
        while True:
            with self.lock:
                now = time.time()
                q = self.events[model]
                while q and now - q[0][0] > 60:
                    q.popleft()
                used = sum(t for _, t in q)
                if used + est_tokens <= self.tpm or not q:
                    q.append((now, est_tokens))
                    return
                sleep_for = 60 - (now - q[0][0]) + 0.5
            time.sleep(max(sleep_for, 0.5))

    def correct(self, model, est, actual):
        with self.lock:
            q = self.events[model]
            for i in range(len(q) - 1, -1, -1):
                if q[i][1] == est:
                    q[i] = (q[i][0], actual)
                    break


THROTTLE = _Throttle(config.TOKENS_PER_MINUTE)


def _estimate_tokens(messages, max_tokens):
    chars = sum(len(m["content"]) for m in messages)
    # completions are typically ~40% of max_tokens; a 429 is retried anyway
    return int(chars / 4.0) + int(0.45 * max_tokens)


def _cache_path(payload) -> Path:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return config.CACHE_DIR / (hashlib.sha256(blob.encode("utf-8")).hexdigest() + ".json")


def chat(messages, role, model=None, max_tokens=1500, json_mode=False, temperature=None):
    """Return the assistant text for `messages`. Cached, throttled, retried."""
    model = model or config.MODELS[role]
    temperature = config.TEMPERATURE if temperature is None else temperature
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": config.SEED,
    }
    if model.startswith("openai/gpt-oss"):
        payload["reasoning_effort"] = "low"
    elif model.startswith("qwen/"):
        payload["reasoning_effort"] = "none"
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    cache_file = _cache_path(payload)
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        USAGE.add(role, model, data.get("usage", {}), data.get("latency", 0.0), cached=True)
        return data["content"]

    if EXHAUSTED.get(model, 0) > time.time():
        raise DailyLimitReached(f"{model}: daily quota exhausted (cached)")
    headers = {"Authorization": f"Bearer {_load_api_key()}"}
    est = _estimate_tokens(messages, max_tokens)
    last_err = None
    for attempt in range(8):
        THROTTLE.wait(model, est)
        t0 = time.time()
        try:
            r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=180)
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(5 * (attempt + 1))
            continue
        latency = time.time() - t0

        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage", {})
            THROTTLE.correct(model, est, usage.get("total_tokens", est))
            if json_mode and not content.strip():
                last_err = RuntimeError("empty JSON response")
                payload["max_tokens"] = int(payload["max_tokens"] * 1.5)
                continue
            cache_file.write_text(json.dumps(
                {"content": content, "usage": usage, "latency": latency, "model": model,
                 "role": role}, ensure_ascii=False), encoding="utf-8")
            USAGE.add(role, model, usage, latency, cached=False)
            return content

        text = r.text
        if r.status_code == 429:
            if "per day" in text or "TPD" in text or "RPD" in text:
                wait = _retry_after(r, text)
                if wait is not None and wait <= 120:
                    time.sleep(wait + 1)
                    continue
                EXHAUSTED[model] = time.time() + max(600, wait or 0)
                raise DailyLimitReached(f"{model}: {text[:300]}")
            time.sleep((_retry_after(r, text) or 10) + 1)
            continue
        if r.status_code == 400 and "json_validate_failed" in text and json_mode:
            # Model produced invalid JSON: retry once without strict JSON mode.
            payload.pop("response_format", None)
            last_err = RuntimeError(text[:300])
            continue
        if r.status_code == 413 or "reduce your message size" in text:
            raise RuntimeError(f"Request too large for {model}: {text[:300]}")
        last_err = RuntimeError(f"HTTP {r.status_code}: {text[:300]}")
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries ({role}/{model}): {last_err}")


def _retry_after(response, text):
    ra = response.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", text)
    if m:
        h, mi, s = m.groups()
        return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s)
    m = re.search(r"try again in ([\d.]+)ms", text)
    if m:
        return float(m.group(1)) / 1000
    return None


def parse_json(text):
    """Robustly extract a JSON object/array from model output."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = text.find(open_c)
        end = text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def chat_json(messages, role, model=None, max_tokens=1500, default=None):
    out = chat(messages, role, model=model, max_tokens=max_tokens, json_mode=True)
    parsed = parse_json(out)
    return parsed if parsed is not None else default
