from __future__ import annotations

from typing import Any


def _proposal_source(proposal: dict[str, Any]) -> dict[str, str]:
    source = proposal.get("source")
    if not isinstance(source, dict):
        return {"provider": "heuristic", "model": ""}
    provider = str(source.get("provider") or "").strip() or ("openai" if source.get("llm_used") else "heuristic")
    model = str(source.get("model") or "").strip()
    return {"provider": provider, "model": model}


def _risk_labels(context_pack: dict[str, Any] | None) -> dict[str, str]:
    radar = (context_pack or {}).get("radar") if isinstance(context_pack, dict) else None
    radar = radar if isinstance(radar, dict) else {}
    return {
        "energy": str(radar.get("energy_inferred") or "unknown"),
        "deadline": str(radar.get("deadline_risk") or "unknown"),
        "inventory": str(radar.get("inventory_risk") or "unknown"),
    }


def _critique_for(archetype: str, *, risks: dict[str, str]) -> list[str]:
    """
    Deterministic “debate” blurbs that make the tradeoffs explicit for judges.
    Keep it short and planning-focused.
    """
    energy = (risks.get("energy") or "unknown").lower()
    deadline = (risks.get("deadline") or "unknown").lower()
    inventory = (risks.get("inventory") or "unknown").lower()

    if archetype == "sleep_first":
        out = [
            "Critique of Errands-first: energy + transitions can spiral when you’re already tired.",
            "Critique of Admin-first: easy to overrun and still feel “behind” without recovery.",
        ]
        if inventory in {"high", "medium"}:
            out.append("Exception: if supplies are urgent, keep it to delivery/pickup or one-stop only.")
        return out[:3]

    if archetype == "errands_first":
        out = [
            "Critique of Rest-first: tomorrow breaks if you run out of essentials.",
            "Critique of Admin-first: admin can wait if it doesn’t unblock today’s basics.",
        ]
        if energy == "low":
            out.append("Guardrail: one loop, one store, no browsing; bail early if it’s dragging.")
        return out[:3]

    if archetype == "admin_first":
        out = [
            "Critique of Rest-first: deadline anxiety grows if you don’t start the next step.",
            "Critique of Errands-first: errands can consume the whole window and still miss the form/email.",
        ]
        if deadline in {"high", "medium"}:
            out.append("Guardrail: set a hard stop time before you start; finish the first concrete step only.")
        return out[:3]

    return []


def build_debate_cards(
    *,
    proposals: list[dict[str, Any]],
    decision: dict[str, Any] | None,
    context_pack: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    risks = _risk_labels(context_pack)
    selected = str((decision or {}).get("selected_archetype") or "")

    cards: list[dict[str, Any]] = []
    for p in proposals:
        archetype = str(p.get("archetype") or "")
        if not archetype:
            continue
        source = _proposal_source(p)
        cards.append(
            {
                "archetype": archetype,
                "title": str(p.get("title") or archetype.replace("_", "-")).strip(),
                "selected": archetype == selected,
                "provider": source["provider"],
                "model": source["model"],
                "start_here": list((p.get("start_here") or [])[:3]) if isinstance(p.get("start_here"), list) else [],
                "pitch": list((p.get("rationale") or [])[:2]) if isinstance(p.get("rationale"), list) else [],
                "tradeoffs": list((p.get("tradeoffs") or [])[:2]) if isinstance(p.get("tradeoffs"), list) else [],
                "critique": _critique_for(archetype, risks=risks),
            }
        )
    # Stable ordering for the UI.
    order = {"sleep_first": 0, "errands_first": 1, "admin_first": 2}
    cards.sort(key=lambda c: order.get(str(c.get("archetype")), 99))
    return cards

