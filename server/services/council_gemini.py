from __future__ import annotations

from typing import Any, Literal

from server.services.gemini_generate import GeminiError, extract_text, generate_content
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


def generate_proposal_gemini(
    *,
    api_key: str,
    model: str,
    archetype: Archetype,
    logs: list[dict],
    energy: str | None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    resp = generate_content(
        api_key=api_key,
        model=model,
        user_text=_proposal_prompt(archetype=archetype, logs=logs, energy=energy),
        timeout_s=timeout_s,
    )
    text = extract_text(resp)
    if not text:
        raise GeminiError("Empty model response")
    try:
        return parse_json_object(text)
    except LLMJSONError as e:
        raise GeminiError(str(e)) from e

