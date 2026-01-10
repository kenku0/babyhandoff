from __future__ import annotations

from typing import Any, Literal

from server.services.anthropic_messages import AnthropicError, extract_text, messages_create
from server.services.llm_json import LLMJSONError, parse_json_object


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
    title_hint = {
        "sleep_first": "Stabilize-first (minimum viable shift)",
        "errands_first": "Errands-first (batch + reduce supply risk)",
        "admin_first": "Admin-first (deadlines + logistics)",
    }.get(archetype, "")
    return f"""You are a planning assistant. Create a concise, practical plan for the next 12 hours.

Constraints:
- This is NOT medical advice. Avoid diagnosis/treatment, meds, or clinical recommendations.
- Keep it minimal: 3 plan blocks, 3–5 top priorities, 2–3 rationale bullets.
- Archetype: {archetype.replace("_", "-")}
- Energy override: {energy or "auto"}
- Suggested title: {title_hint}

Notes:
{_logs_to_bullets(logs)}

Return ONLY valid JSON with keys:
{{
  "title": string,
  "start_here": string[],
  "plan_blocks": string[],
  "top_priorities": string[],
  "rationale": string[],
  "tradeoffs": string[],
  "stop_rule": string
}}
"""


def generate_proposal_anthropic(
    *,
    api_key: str,
    model: str,
    archetype: Archetype,
    logs: list[dict],
    energy: str | None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    resp = messages_create(
        api_key=api_key,
        model=model,
        user_text=_proposal_prompt(archetype=archetype, logs=logs, energy=energy),
        timeout_s=timeout_s,
    )
    text = extract_text(resp)
    if not text:
        raise AnthropicError("Empty model response")
    try:
        return parse_json_object(text)
    except LLMJSONError as e:
        raise AnthropicError(str(e)) from e
