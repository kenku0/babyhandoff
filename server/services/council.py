from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, Literal

from server.agents.risk_radar import infer_deadline_risk, infer_energy, infer_inventory_risk


Archetype = Literal["sleep_first", "errands_first", "admin_first"]
EnergyLevel = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class CouncilOutput:
    proposals: list[dict[str, Any]]
    decision: dict[str, Any]
    handoff_markdown: str


def _normalize_energy(energy_override: str | None, logs: list[dict]) -> EnergyLevel | None:
    if energy_override in {"high", "medium", "low"}:
        return energy_override  # type: ignore[return-value]
    texts = [str(l.get("text") or "") for l in logs]
    return infer_energy(texts)


def _top_evidence(logs: list[dict], *, log_type: str, limit: int = 2) -> list[dict]:
    matches = [l for l in logs if l.get("type") == log_type]
    out: list[dict] = []
    for l in matches[:limit]:
        out.append(
            {
                "log_id": str(l.get("_id")),
                "type": l.get("type"),
                "text": l.get("text"),
                "created_at": l.get("created_at"),
            }
        )
    return out


def _score_proposal(
    archetype: Archetype,
    *,
    energy: EnergyLevel | None,
    deadline_risk: str,
    inventory_risk: str,
) -> dict[str, Any]:
    scores: dict[str, Any] = {}

    scores["commitments_fit"] = {"score": 3, "why": "No calendar integration in MVP; assume neutral fit."}

    if deadline_risk == "high":
        deadline = 5 if archetype == "admin_first" else 2 if archetype == "sleep_first" else 3
    elif deadline_risk == "medium":
        deadline = 4 if archetype == "admin_first" else 3
    else:
        deadline = 3
    scores["deadline_risk"] = {"score": deadline, "why": f"Deadline risk inferred as {deadline_risk}."}

    if energy == "low":
        energy_score = 5 if archetype == "sleep_first" else 2 if archetype == "errands_first" else 3
    elif energy == "high":
        energy_score = 4 if archetype == "errands_first" else 3
    else:
        energy_score = 3
    scores["energy_match"] = {"score": energy_score, "why": f"Energy inferred as {energy or 'unknown'}."}

    time_window = 4 if archetype in {"sleep_first", "admin_first"} else 3
    if energy == "high" and archetype == "errands_first":
        time_window = 4
    if energy == "low" and archetype == "errands_first":
        time_window = 2
    scores["time_window"] = {"score": time_window, "why": "Based on energy and archetype assumptions."}

    if inventory_risk in {"high", "medium"}:
        inv = 5 if archetype == "errands_first" else 3
    else:
        inv = 3
    scores["inventory_risk"] = {"score": inv, "why": f"Inventory risk inferred as {inventory_risk}."}

    total = sum(int(v["score"]) for v in scores.values())
    scores["_total"] = {"score": total, "why": "Sum of rubric scores."}
    return scores


def _proposal_sleep_first(*, energy: EnergyLevel | None, deadline_risk: str, inventory_risk: str) -> dict[str, Any]:
    blocks = ["Rest block (45–90 min)", "One admin block (15–30 min)", "One essential task only"]
    if deadline_risk in {"high", "medium"}:
        blocks[1] = "Admin block (30–45 min) to clear nearest deadline"
    if inventory_risk in {"high", "medium"}:
        blocks[2] = "Single-stop supply run (diapers/wipes/cream) — keep it short"
    top = ["Protect one rest block", "Clear the nearest deadline", "Cover essentials (supplies/food)"]
    rationale = [
        "You’re operating under fatigue; reduce cognitive load and avoid over-committing.",
        "Handle the one admin item that can cascade into stress.",
    ]
    if energy == "low":
        rationale.append("Energy is low; prioritize recovery to prevent a worse day tomorrow.")
    return {"archetype": "sleep_first", "title": "Sleep-first", "plan_blocks": blocks, "top_priorities": top, "rationale": rationale}


def _proposal_errands_first(*, energy: EnergyLevel | None, inventory_risk: str) -> dict[str, Any]:
    blocks = ["One errands window (45–90 min, one loop)", "Quick admin block (15–30 min)", "Minimal home reset (15 min)"]
    if inventory_risk in {"high", "medium"}:
        blocks[0] = "One errands window (45–75 min): supplies first, groceries second"
    top = ["Buy essentials (supplies + simple food)", "Prevent a supply emergency tomorrow", "Knock out one admin task if time allows"]
    rationale = [
        "Batching errands reduces overhead and prevents running out at a bad time.",
        "One loop beats three separate trips.",
    ]
    if energy == "low":
        rationale.append("Energy is low; keep the loop to one stop if possible.")
    return {"archetype": "errands_first", "title": "Errands-first", "plan_blocks": blocks, "top_priorities": top, "rationale": rationale}


def _proposal_admin_first(*, energy: EnergyLevel | None, deadline_risk: str, inventory_risk: str) -> dict[str, Any]:
    blocks = ["Admin sprint (30–60 min) to clear deadlines", "Focus block (60–90 min)", "One essential task only"]
    if deadline_risk == "none":
        blocks[0] = "Admin sweep (20–30 min): schedule, email, quick forms"
    if energy == "low":
        blocks[1] = "Short focus block (30–45 min) + stop"
    top = ["Remove deadline anxiety", "Protect a small focus block", "Do one essential supply/food task"]
    rationale = [
        "Admin tasks are easy to miss and high stress when late.",
        "A small focus block prevents school/work from silently slipping.",
    ]
    if inventory_risk in {"high", "medium"}:
        rationale.append("Supplies are trending low; include one essential pickup.")
    return {"archetype": "admin_first", "title": "Admin-first", "plan_blocks": blocks, "top_priorities": top, "rationale": rationale}


def merge_llm_proposal(base: dict[str, Any], llm: dict[str, Any] | None) -> dict[str, Any]:
    if not llm:
        return base
    merged = dict(base)
    if isinstance(llm.get("title"), str) and llm["title"].strip():
        merged["title"] = llm["title"].strip()
    for key in ["plan_blocks", "top_priorities", "rationale"]:
        value = llm.get(key)
        if isinstance(value, list) and any(isinstance(x, str) and x.strip() for x in value):
            merged[key] = [str(x).strip() for x in value if isinstance(x, str) and x.strip()]
    return merged


def _make_markdown(
    *,
    selected: dict[str, Any],
    energy: EnergyLevel | None,
    deadline_risk: str,
    inventory_risk: str,
    logs: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# BabyHandoff — Shift Handoff")
    lines.append("")
    lines.append(f"## Selected Plan: {selected['title']}")
    lines.append("")
    lines.append("### Why this plan won")
    for r in (selected.get("rationale") or [])[:3]:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("## Next 12h Plan (blocks)")
    for b in selected.get("plan_blocks") or []:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("## Top Priorities")
    for p in (selected.get("top_priorities") or [])[:5]:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## 24h Outlook (lightweight)")
    lines.append(f"- Deadline risk: **{deadline_risk}**")
    lines.append(f"- Inventory risk: **{inventory_risk}**")
    lines.append(f"- Energy: **{energy or 'unknown'}**")
    lines.append("- Prep one thing now to make tomorrow easier (admin/supplies/rest).")
    lines.append("")
    lines.append("## Notes (raw)")
    for l in logs[-12:]:
        lines.append(f"- [{str(l.get('type') or '').upper()}] {str(l.get('text') or '').strip()}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def unified_diff(prev_markdown: str, next_markdown: str) -> str:
    diff = difflib.unified_diff(
        prev_markdown.splitlines(True),
        next_markdown.splitlines(True),
        fromfile="v1",
        tofile="v2",
    )
    return "".join(diff)


def proposal_to_markdown(proposal: dict[str, Any]) -> str:
    title = str(proposal.get("title") or "Proposal").strip()
    archetype = str(proposal.get("archetype") or "").strip()
    lines: list[str] = []
    lines.append(f"### {title}" + (f" ({archetype.replace('_', '-')})" if archetype else ""))
    lines.append("")
    lines.append("**Blocks**")
    for b in proposal.get("plan_blocks") or []:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("**Top priorities**")
    for p in proposal.get("top_priorities") or []:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("**Rationale**")
    for r in proposal.get("rationale") or []:
        lines.append(f"- {r}")
    return "\n".join(lines).strip() + "\n"


def decision_to_markdown(decision: dict[str, Any]) -> str:
    selected = str(decision.get("selected_archetype") or "").strip()
    summary = str(decision.get("summary") or "").strip()
    lines: list[str] = []
    lines.append(f"### Decision: {selected.replace('_', '-')}" if selected else "### Decision")
    lines.append("")
    if summary:
        lines.append(summary)
        lines.append("")
    lines.append("**Scores (total / deadline / energy / inventory)**")
    scores = decision.get("scores") or {}
    for archetype, s in scores.items():
        try:
            total = s["_total"]["score"]
            dl = s["deadline_risk"]["score"]
            en = s["energy_match"]["score"]
            inv = s["inventory_risk"]["score"]
            lines.append(f"- {archetype.replace('_', '-')}: {total} / {dl} / {en} / {inv}")
        except Exception:
            lines.append(f"- {str(archetype).replace('_', '-')}: (unavailable)")
    evidence = decision.get("evidence_links") or []
    if evidence:
        lines.append("")
        lines.append("**Evidence excerpts**")
        for e in evidence:
            t = str(e.get("type") or "note").upper()
            txt = str(e.get("text") or "").strip()
            if txt:
                lines.append(f"- [{t}] {txt}")
    return "\n".join(lines).strip() + "\n"


def build_council_transcript_markdown(
    *,
    shift_id: str,
    run_id: str,
    council_mode: str,
    energy_override: str | None,
    proposals: list[dict[str, Any]],
    decision: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# BabyHandoff — Council Transcript")
    lines.append("")
    lines.append("_BabyHandoff is a planning tool. It does not provide medical advice._")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- Shift: `{shift_id}`")
    lines.append(f"- Run: `{run_id}`")
    lines.append(f"- Council mode: `{council_mode}`")
    lines.append(f"- Energy override: `{energy_override or 'auto'}`")
    lines.append("")
    lines.append("## Proposals")
    lines.append("")
    for p in proposals:
        lines.append(proposal_to_markdown(p).rstrip())
        lines.append("")
    lines.append(decision_to_markdown(decision).rstrip())
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_council_transcript_markdown_from_messages(*, shift_id: str, run_id: str, messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# BabyHandoff — Council Transcript")
    lines.append("")
    lines.append("_BabyHandoff is a planning tool. It does not provide medical advice._")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- Shift: `{shift_id}`")
    lines.append(f"- Run: `{run_id}`")
    lines.append("")
    for m in messages:
        sender = str(m.get("sender") or "System")
        created_at = m.get("created_at")
        when = str(created_at) if created_at else ""
        role = str(m.get("role") or "").strip()
        header = f"## {sender}" + (f" · {when}" if when else "") + (f" · {role}" if role else "")
        lines.append(header)
        lines.append("")
        content = str(m.get("content") or "").strip()
        lines.append(content if content else "_No content._")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def run_council(
    *,
    shift_id: str,
    logs: list[dict],
    energy_override: str | None,
    proposal_overrides: dict[str, dict[str, Any]] | None = None,
) -> CouncilOutput:
    energy = _normalize_energy(energy_override, logs)
    deadline_risk = infer_deadline_risk(logs)
    inventory_risk = infer_inventory_risk(logs)

    base_proposals = [
        _proposal_sleep_first(energy=energy, deadline_risk=deadline_risk, inventory_risk=inventory_risk),
        _proposal_errands_first(energy=energy, inventory_risk=inventory_risk),
        _proposal_admin_first(energy=energy, deadline_risk=deadline_risk, inventory_risk=inventory_risk),
    ]
    proposals = (
        [merge_llm_proposal(p, proposal_overrides.get(p["archetype"])) for p in base_proposals]
        if proposal_overrides
        else base_proposals
    )

    score_map: dict[str, Any] = {}
    scored: list[tuple[int, dict[str, Any]]] = []
    for p in proposals:
        archetype: Archetype = p["archetype"]
        scores = _score_proposal(archetype, energy=energy, deadline_risk=deadline_risk, inventory_risk=inventory_risk)
        score_map[archetype] = scores
        scored.append((int(scores["_total"]["score"]), p))

    scored.sort(key=lambda t: t[0], reverse=True)
    winner = scored[0][1]

    evidence: list[dict[str, Any]] = []
    evidence.extend(_top_evidence(logs, log_type="deadline"))
    evidence.extend(_top_evidence(logs, log_type="inventory"))
    evidence.extend(_top_evidence(logs, log_type="note"))

    decision = {
        "selected_archetype": winner["archetype"],
        "summary": f"Selected {winner['title']} based on rubric scoring.",
        "scores": score_map,
        "evidence_links": evidence[:6],
    }

    handoff_md = _make_markdown(
        selected=winner,
        energy=energy,
        deadline_risk=deadline_risk,
        inventory_risk=inventory_risk,
        logs=logs,
    )
    return CouncilOutput(proposals=proposals, decision=decision, handoff_markdown=handoff_md)
