from __future__ import annotations

import json
from typing import Any, Literal

from server.services.openai_responses import OpenAIError, extract_text, responses_create


Archetype = Literal["sleep_first", "errands_first", "admin_first"]


def _logs_to_bullets(logs: list[dict], limit: int = 40) -> str:
    lines: list[str] = []
    for l in logs[-limit:]:
        t = str(l.get("type") or "note").upper()
        txt = str(l.get("text") or "").strip()
        if not txt:
            continue
        lines.append(f"- [{t}] {txt}")
    return "\n".join(lines)


def _proposal_prompt(*, archetype: Archetype, logs: list[dict], energy: str | None) -> str:
    return f"""You are a planning assistant. Create a concise, practical plan for the next 12 hours.

Constraints:
- This is NOT medical advice. Avoid diagnosis/treatment, meds, or clinical recommendations.
- Keep it minimal: 3 plan blocks, 3–5 top priorities, 2–3 rationale bullets.
- Archetype: {archetype.replace("_", "-")}
- Energy override: {energy or "auto"}

Notes:
{_logs_to_bullets(logs)}

Return ONLY valid JSON with keys:
{{
  "title": string,
  "plan_blocks": string[],
  "top_priorities": string[],
  "rationale": string[]
}}
"""


def generate_proposal_openai(
    *,
    api_key: str,
    model: str,
    archetype: Archetype,
    logs: list[dict],
    energy: str | None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    resp = responses_create(
        api_key=api_key,
        model=model,
        input_text=_proposal_prompt(archetype=archetype, logs=logs, energy=energy),
        timeout_s=timeout_s,
    )
    text = extract_text(resp)
    if not text:
        raise OpenAIError("Empty model response")
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise OpenAIError("Model response was not a JSON object")
    return parsed
