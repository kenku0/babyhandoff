from __future__ import annotations

from typing import Any, Literal

from server.services.llm_json import LLMJSONError, parse_json_object
from server.services.openai_responses import OpenAIError, extract_text, responses_create


Archetype = Literal["sleep_first", "errands_first", "admin_first"]

RUBRIC_DIMENSIONS: list[str] = [
    "commitments_fit",
    "deadline_risk",
    "energy_match",
    "time_window",
    "inventory_risk",
]


def _referee_prompt(
    *,
    proposals: list[dict[str, Any]],
    context_pack_summary: str,
    log_refs: list[dict[str, Any]],
) -> str:
    proposal_min = [
        {
            "archetype": p.get("archetype"),
            "title": p.get("title"),
            "plan_blocks": p.get("plan_blocks"),
            "top_priorities": p.get("top_priorities"),
            "rationale": p.get("rationale"),
        }
        for p in proposals
    ]
    # Keep log refs compact to reduce prompt bloat.
    log_min = [
        {"log_id": r.get("log_id"), "type": r.get("type"), "text": r.get("text")}
        for r in (log_refs or [])[:20]
        if isinstance(r, dict)
    ]
    return "\n".join(
        [
            "You are a referee for a planning app. Choose the best plan archetype for the next 12 hours.",
            "",
            "Safety:",
            "- This is NOT medical advice. Avoid diagnosis/treatment, meds, or clinical recommendations.",
            "",
            "Rubric (score each 1–5):",
            "- commitments_fit: fit with fixed commitments (neutral if unknown)",
            "- deadline_risk: reduces deadline risk",
            "- energy_match: matches current energy",
            "- time_window: feasible within available blocks/travel assumptions",
            "- inventory_risk: reduces 'running low' risk",
            "",
            "Scoring rules:",
            "- Score every rubric dimension for every proposal.",
            "- Total score is the sum of the five dimensions (compute it).",
            "- Pick the highest total. Break ties by: deadline_risk, then inventory_risk.",
            "",
            "Context pack summary:",
            context_pack_summary.strip() or "(empty)",
            "",
            "Evidence refs (cite by log_id):",
            str(log_min),
            "",
            "Proposals:",
            str(proposal_min),
            "",
            "Return ONLY valid JSON with keys:",
            "{",
            '  "selected_archetype": "sleep_first" | "errands_first" | "admin_first",',
            '  "summary": string,',
            '  "scores": {',
            '    "sleep_first": {"commitments_fit": {"score": 1-5, "why": string}, ... },',
            '    "errands_first": {...},',
            '    "admin_first": {...}',
            "  },",
            '  "evidence_links": [{"log_id": string, "type": string, "text": string}]',
            "}",
        ]
    )


def _coerce_score(value: Any) -> int | None:
    try:
        score_int = int(value)
    except Exception:
        return None
    if 1 <= score_int <= 5:
        return score_int
    return None


def score_decision_openai(
    *,
    api_key: str,
    model: str,
    proposals: list[dict[str, Any]],
    context_pack: dict[str, Any] | None,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    prompt = _referee_prompt(
        proposals=proposals,
        context_pack_summary=str((context_pack or {}).get("summary") or ""),
        log_refs=list((context_pack or {}).get("log_refs") or []),
    )
    resp = responses_create(api_key=api_key, model=model, input_text=prompt, timeout_s=timeout_s)
    text = extract_text(resp)
    if not text:
        raise OpenAIError("Empty model response")
    try:
        parsed = parse_json_object(text)
    except LLMJSONError as e:
        raise OpenAIError(str(e)) from e

    selected = str(parsed.get("selected_archetype") or "").strip()
    ordered_archetypes: list[str] = []
    for p in proposals:
        a = str(p.get("archetype") or "").strip()
        if a and a not in ordered_archetypes:
            ordered_archetypes.append(a)
    valid_archetypes = set(ordered_archetypes)
    if selected not in valid_archetypes:
        raise OpenAIError(f"Invalid selected_archetype: {selected!r}")

    scores_in = parsed.get("scores")
    if not isinstance(scores_in, dict):
        raise OpenAIError("Missing/invalid scores")

    scores_out: dict[str, Any] = {}
    for archetype in ordered_archetypes:
        raw = scores_in.get(archetype)
        if not isinstance(raw, dict):
            raise OpenAIError(f"Missing score block for {archetype}")
        dims: dict[str, Any] = {}
        total = 0
        for d in RUBRIC_DIMENSIONS:
            entry = raw.get(d) if isinstance(raw, dict) else None
            if not isinstance(entry, dict):
                raise OpenAIError(f"Missing {d} score for {archetype}")
            score = _coerce_score(entry.get("score"))
            if score is None:
                raise OpenAIError(f"Invalid {d}.score for {archetype}")
            why = str(entry.get("why") or "").strip()[:240]
            dims[d] = {"score": score, "why": why}
            total += score
        dims["_total"] = {"score": total, "why": "Sum of rubric scores."}
        scores_out[archetype] = dims

    evidence_links_in = parsed.get("evidence_links") or []
    evidence_links: list[dict[str, Any]] = []
    if isinstance(evidence_links_in, list):
        for e in evidence_links_in[:8]:
            if not isinstance(e, dict):
                continue
            log_id = str(e.get("log_id") or "").strip()
            t = str(e.get("type") or "").strip()
            txt = str(e.get("text") or "").strip()
            if log_id and txt:
                evidence_links.append({"log_id": log_id, "type": t, "text": txt})

    return {
        "selected_archetype": selected,
        "summary": str(parsed.get("summary") or "").strip()[:500],
        "scores": scores_out,
        "evidence_links": evidence_links,
        "judge_mode": "openai",
        "judge_model": model,
    }
