from __future__ import annotations

import re
from typing import Any, Literal

from server.services.anthropic_messages import AnthropicError, extract_text as anthropic_extract_text, messages_create
from server.services.gemini_generate import GeminiError, extract_text as gemini_extract_text, generate_content
from server.services.openai_responses import OpenAIError, extract_text as openai_extract_text, responses_create


Provider = Literal["openai", "anthropic", "gemini", "heuristic"]
Archetype = Literal["sleep_first", "errands_first", "admin_first"]
Energy = Literal["high", "medium", "low"]


def _safe_line(s: Any) -> str:
    return str(s or "").replace("\n", " ").strip()


def _proposal_digest(proposals: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for p in proposals:
        a = _safe_line(p.get("archetype"))
        title = _safe_line(p.get("title"))
        lines.append(f"- {a}: {title}")
        blocks = p.get("plan_blocks") or []
        if isinstance(blocks, list):
            for b in blocks[:3]:
                if isinstance(b, str) and b.strip():
                    lines.append(f"  - {b.strip()}")
        tradeoffs = p.get("tradeoffs") or []
        if isinstance(tradeoffs, list) and any(isinstance(x, str) and x.strip() for x in tradeoffs):
            lines.append("  - tradeoffs:")
            for t in tradeoffs[:2]:
                if isinstance(t, str) and t.strip():
                    lines.append(f"    - {t.strip()}")
    return "\n".join(lines).strip()


def build_roundtable_prompt(
    *,
    proposals: list[dict[str, Any]],
    selected_archetype: str,
    energy: str | None,
    deadline_risk: str,
    inventory_risk: str,
) -> str:
    return f"""You are a strict, practical reviewer of short-term plans.

Constraints:
- This is NOT medical advice. Avoid diagnosis/treatment, meds, or clinical recommendations.
- Be concise. No fluff.

Context:
- Energy: {energy or "unknown"}
- Deadline risk: {deadline_risk}
- Inventory risk: {inventory_risk}
- Current selected plan (by rubric): {selected_archetype}

Plans:
{_proposal_digest(proposals)}

Return plain text with this exact structure:
VOTE: <sleep_first|errands_first|admin_first>
WHY:
- <reason 1>
- <reason 2>
CONCERN: <one short concern about the selected plan>
TWEAK: <one actionable improvement to the selected plan>
"""


_VOTE_RE = re.compile(r"^\s*VOTE\s*:\s*(sleep_first|errands_first|admin_first)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_vote(text: str) -> Archetype | None:
    m = _VOTE_RE.search(text or "")
    if not m:
        return None
    return m.group(1).lower()  # type: ignore[return-value]


def _heuristic_votes(
    *,
    energy: str | None,
    deadline_risk: str,
    inventory_risk: str,
) -> list[tuple[str, Archetype, str]]:
    votes: list[tuple[str, Archetype, str]] = []
    if energy == "low":
        votes.append(("EnergyCritic", "sleep_first", "Low energy → stabilize execution first."))
    else:
        votes.append(("EnergyCritic", "errands_first", "Energy ok → batch errands to reduce future friction."))

    if deadline_risk in {"high", "medium"}:
        votes.append(("DeadlineCritic", "admin_first", "Deadlines cascade stress → clear the nearest one."))
    else:
        votes.append(("DeadlineCritic", "sleep_first", "No urgent deadline → protect recovery and reduce mistakes."))

    if inventory_risk in {"high", "medium"}:
        votes.append(("InventoryCritic", "errands_first", "Low supplies → prevent a 2am emergency run."))
    else:
        votes.append(("InventoryCritic", "admin_first", "Supplies ok → use focus window for admin + planning."))

    return votes[:3]


def generate_roundtable_vote(
    *,
    provider: Provider,
    api_key: str,
    model: str,
    proposals: list[dict[str, Any]],
    selected_archetype: str,
    energy: str | None,
    deadline_risk: str,
    inventory_risk: str,
    timeout_s: float = 20.0,
) -> str:
    prompt = build_roundtable_prompt(
        proposals=proposals,
        selected_archetype=selected_archetype,
        energy=energy,
        deadline_risk=deadline_risk,
        inventory_risk=inventory_risk,
    )
    if provider == "openai":
        resp = responses_create(api_key=api_key, model=model, input_text=prompt, timeout_s=timeout_s)
        text = openai_extract_text(resp)
        if not text:
            raise OpenAIError("Empty model response")
        return text.strip()
    if provider == "anthropic":
        resp = messages_create(api_key=api_key, model=model, user_text=prompt, timeout_s=timeout_s, max_tokens=450)
        text = anthropic_extract_text(resp)
        if not text:
            raise AnthropicError("Empty model response")
        return text.strip()
    if provider == "gemini":
        resp = generate_content(api_key=api_key, model=model, user_text=prompt, timeout_s=timeout_s, max_output_tokens=450)
        text = gemini_extract_text(resp)
        if not text:
            raise GeminiError("Empty model response")
        return text.strip()
    raise ValueError(f"Unsupported provider: {provider}")


def synthesize_roundtable_votes(
    *,
    proposals: list[dict[str, Any]],
    energy: str | None,
    deadline_risk: str,
    inventory_risk: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for persona, vote, reason in _heuristic_votes(
        energy=energy, deadline_risk=deadline_risk, inventory_risk=inventory_risk
    ):
        out.append(
            {
                "provider": "heuristic",
                "model": persona,
                "vote": vote,
                "content": "\n".join(
                    [
                        f"VOTE: {vote}",
                        "WHY:",
                        f"- {reason}",
                        "- Weighted toward reducing cascading failures.",
                        "CONCERN: The selected plan may still feel too big when fatigue spikes.",
                        "TWEAK: Time-box the first block and set a hard stop before adding anything else.",
                    ]
                ),
            }
        )
    return out

