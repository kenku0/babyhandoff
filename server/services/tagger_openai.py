from __future__ import annotations

from typing import Any

from server.services.llm_json import LLMJSONError, parse_json_object
from server.services.openai_responses import OpenAIError, extract_text, responses_create


ALLOWED_TAGS = ["note", "task", "deadline", "inventory"]


def _tagger_prompt(*, text: str) -> str:
    return f"""You label messy parent notes into structured items.

Allowed tags (choose 1+ per item):
- note: general context or feelings
- task: an action to do
- deadline: anything time-sensitive ("today", "by tomorrow 5pm", etc.)
- inventory: supplies running low / need to buy

Instructions:
- Split the input into 1–6 items if it contains multiple distinct points.
- Each item should be a short rewritten sentence preserving the user's meaning.
- Each item must have 1–3 tags from the allowed list.
- Return ONLY JSON.

Input:
{text}

Return JSON:
{{
  "items": [
    {{"text": string, "tags": string[]}}
  ]
}}
"""


def tag_items_openai(*, api_key: str, model: str, text: str, timeout_s: float = 10.0) -> list[dict[str, Any]]:
    resp = responses_create(api_key=api_key, model=model, input_text=_tagger_prompt(text=text), timeout_s=timeout_s)
    raw = extract_text(resp)
    if not raw:
        raise OpenAIError("Empty model response")
    try:
        obj = parse_json_object(raw)
    except LLMJSONError as e:
        raise OpenAIError(str(e)) from e

    items = obj.get("items")
    if not isinstance(items, list) or not items:
        raise OpenAIError("Tagger returned no items")

    cleaned: list[dict[str, Any]] = []
    for it in items[:6]:
        if not isinstance(it, dict):
            continue
        item_text = str(it.get("text") or "").strip()
        tags = it.get("tags")
        if not item_text:
            continue
        if not isinstance(tags, list):
            tags = []
        tags_clean = []
        for t in tags:
            t = str(t).strip().lower()
            if t in ALLOWED_TAGS and t not in tags_clean:
                tags_clean.append(t)
        if not tags_clean:
            tags_clean = ["note"]
        cleaned.append({"text": item_text, "tags": tags_clean[:3]})

    if not cleaned:
        raise OpenAIError("Tagger returned no valid items")
    return cleaned


def tag_items_heuristic(text: str) -> list[dict[str, Any]]:
    parts = [p.strip() for p in (text or "").replace("•", "\n").splitlines() if p.strip()]
    if not parts:
        return []
    out: list[dict[str, Any]] = []
    for p in parts[:6]:
        t = p.lower()
        tags: list[str] = []
        if any(k in t for k in ["deadline", "by ", "due", "tomorrow", "today", "tonight", "asap"]):
            tags.append("deadline")
        if any(k in t for k in ["running low", "low on", "almost out", "out of", "wipes", "diapers"]):
            tags.append("inventory")
        if any(k in t for k in ["pick up", "buy", "order", "call", "submit", "schedule", "email", "fill out"]):
            tags.append("task")
        if not tags:
            tags.append("note")
        out.append({"text": p, "tags": tags[:3]})
    return out

