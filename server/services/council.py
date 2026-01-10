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
    matches = [l for l in logs if (l.get("type") == log_type) or (log_type in (l.get("tags") or []))]
    out: list[dict] = []
    for l in matches[:limit]:
        out.append(
            {
                "log_id": str(l.get("_id")),
                "type": l.get("type"),
                "tags": l.get("tags") or [],
                "text": l.get("text"),
                "created_at": l.get("created_at"),
            }
        )
    return out


def _first_log_text(logs: list[dict], *, log_type: str) -> str | None:
    for l in logs:
        if l.get("type") != log_type:
            continue
        txt = str(l.get("text") or "").strip()
        if txt:
            return txt
    return None


def _timeline_totals(timeline: list[dict[str, Any]] | None) -> tuple[int, int]:
    if not timeline:
        return (0, 0)
    min_total = 0
    max_total = 0
    for item in timeline:
        try:
            mn = int(item.get("min") or 0)
        except Exception:
            mn = 0
        try:
            mx = int(item.get("max") or 0)
        except Exception:
            mx = 0
        if mx <= 0:
            mx = mn
        min_total += max(mn, 0)
        max_total += max(mx, 0)
    return (min_total, max_total)


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


def _proposal_sleep_first(
    *,
    energy: EnergyLevel | None,
    deadline_risk: str,
    inventory_risk: str,
    logs: list[dict] | None = None,
) -> dict[str, Any]:
    logs = logs or []
    deadline_text = _first_log_text(logs, log_type="deadline")
    inventory_text = _first_log_text(logs, log_type="inventory")
    task_text = _first_log_text(logs, log_type="task")

    admin_min, admin_max = (10, 20)
    if deadline_risk in {"high", "medium"}:
        admin_min, admin_max = (20, 30)

    supplies_min, supplies_max = (15, 30)
    if inventory_risk in {"high", "medium"}:
        supplies_min, supplies_max = (20, 30)

    blocks = [
        "Recovery block (60–90 min): timer on, phone away, lights low",
        f"Admin micro-sprint ({admin_min}–{admin_max} min): start the nearest deadline (draft/save/submit)",
        "Essentials (15–30 min): supplies/food in one stop (delivery/pickup if possible)",
    ]
    if deadline_risk in {"high", "medium"}:
        blocks[1] = "Admin block (20–30 min): start or finish the nearest deadline"
    if inventory_risk in {"high", "medium"}:
        blocks[2] = f"Single-stop supply coverage ({supplies_min}–{supplies_max} min): order/pickup/one-stop (diapers/wipes/cream)"
    top = [
        "Protect one recovery block (non-negotiable)",
        "Make the nearest deadline safe (first concrete step)",
        "Cover supplies/food so tomorrow doesn’t break",
    ]
    rationale = [
        "Fatigue taxes decision-making; fewer choices beats a perfect plan.",
        "Two small actions now prevent cascade failures (deadline stress + supply emergencies).",
    ]
    if energy == "low":
        rationale.append("Energy is low; recovery makes the rest of the shift feasible.")
    start_here: list[str] = [
        "Set a 60–90 min recovery timer; put your phone across the room.",
        "Write a 1-line rule: “Only 3 things: recover + deadline step + supplies.”",
    ]
    if deadline_text and deadline_risk != "none":
        start_here.append(f"Deadline: “{deadline_text}” → do the first concrete step now (open, fill 3 fields, save draft).")
    if inventory_text and inventory_risk != "none":
        start_here.append(f"Supplies: “{inventory_text}” → add to cart/pickup list now (no browsing).")
    if not (deadline_text or inventory_text) and task_text:
        start_here.append(f"Essentials: “{task_text}” → pick delivery/pickup/one-stop and start it now.")
    stop_rule = "Stop when the deadline is started and supplies are covered; then rest (everything else can wait)."
    wins: list[str] = ["Recovery protected", "Deadline stabilized", "Supplies covered"]
    if deadline_risk == "none":
        wins[1] = "Admin anxiety reduced"
    if inventory_risk == "none":
        wins[2] = "Essentials covered"

    timeline = [
        {"label": "Recovery block", "min": 60, "max": 90, "tag": "recovery"},
        {"label": "Admin sprint (nearest deadline)", "min": admin_min, "max": admin_max, "tag": "admin"},
        {"label": "Supplies/food (one stop)", "min": supplies_min, "max": supplies_max, "tag": "errands"},
    ]
    tmin, tmax = _timeline_totals(timeline)
    if_then = [
        "If energy drops further, do recovery first and shorten everything else to a single 10‑min step.",
        "If supplies are already covered, skip errands and take a second short rest block.",
    ]
    tradeoffs = [
        "Defer non-urgent errands and deep work; keep scope tiny.",
        "If you feel “behind”, write tasks down and re-run later—don’t expand this shift.",
    ]
    return {
        "archetype": "sleep_first",
        "title": "Stability-first",
        "plan_blocks": blocks,
        "timeline": timeline,
        "timeline_total_min": tmin,
        "timeline_total_max": tmax,
        "wins": wins,
        "top_priorities": top,
        "rationale": rationale,
        "start_here": start_here[:5],
        "stop_rule": stop_rule,
        "if_then": if_then,
        "tradeoffs": tradeoffs,
    }


def _proposal_errands_first(*, energy: EnergyLevel | None, inventory_risk: str) -> dict[str, Any]:
    err_min, err_max = (45, 90)
    if inventory_risk in {"high", "medium"}:
        err_min, err_max = (45, 75)
    blocks = [
        f"One errands window ({err_min}–{err_max} min, one loop)",
        "Quick admin block (15–30 min)",
        "Minimal home reset (15 min)",
    ]
    if inventory_risk in {"high", "medium"}:
        blocks[0] = f"One errands window ({err_min}–{err_max} min): supplies first, groceries second"
    top = ["Buy essentials (supplies + simple food)", "Prevent a supply emergency tomorrow", "Knock out one admin task if time allows"]
    rationale = [
        "Batching errands reduces overhead and prevents running out at a bad time.",
        "One loop beats three separate trips.",
    ]
    if energy == "low":
        rationale.append("Energy is low; keep the loop to one stop if possible.")
    start_here = [
        "Pick ONE store/pickup (no browsing).",
        "Write a 60‑second list: diapers/wipes/cream + protein + easy snacks.",
        "Set a hard stop time before you leave.",
    ]
    stop_rule = "Stop after the loop + drop-off. No extra aisles."
    wins = ["Essentials purchased", "Supply risk reduced", "Home reset started"]
    timeline = [
        {"label": "Errands loop (one stop)", "min": err_min, "max": err_max, "tag": "errands"},
        {"label": "Admin quick hit", "min": 15, "max": 30, "tag": "admin"},
        {"label": "Home reset", "min": 15, "max": 15, "tag": "home"},
    ]
    tmin, tmax = _timeline_totals(timeline)
    if_then = [
        "If the store is crowded, switch to pickup/delivery and reclaim 20–30 minutes.",
        "If energy is low, skip the home reset and rest after drop‑off.",
    ]
    tradeoffs = [
        "Recovery likely suffers; schedule a short rest afterward.",
        "Skip anything that adds decision fatigue (browse-y stores, long lists).",
    ]
    return {
        "archetype": "errands_first",
        "title": "Errands-first",
        "plan_blocks": blocks,
        "timeline": timeline,
        "timeline_total_min": tmin,
        "timeline_total_max": tmax,
        "wins": wins,
        "top_priorities": top,
        "rationale": rationale,
        "start_here": start_here,
        "stop_rule": stop_rule,
        "if_then": if_then,
        "tradeoffs": tradeoffs,
    }


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
    start_here = [
        "Open the nearest deadline/form/email thread.",
        "Set a 25–30 min timer and do the first concrete step (draft/save/submit).",
        "Write the 1 essential supply/food action you’ll do later (no more).",
    ]
    stop_rule = "Hard stop at 60 minutes total admin work; then either focus or rest."
    wins = ["Deadline anxiety reduced", "Focus protected", "Essentials covered"]
    focus_min, focus_max = (60, 90)
    if energy == "low":
        focus_min, focus_max = (30, 45)
    admin_min, admin_max = (30, 60) if deadline_risk != "none" else (20, 30)
    essentials_min, essentials_max = (15, 30)
    timeline = [
        {"label": "Admin sprint (deadlines/logistics)", "min": admin_min, "max": admin_max, "tag": "admin"},
        {"label": "Focus block (school/work)", "min": focus_min, "max": focus_max, "tag": "focus"},
        {"label": "Essentials (supplies/food)", "min": essentials_min, "max": essentials_max, "tag": "errands"},
    ]
    tmin, tmax = _timeline_totals(timeline)
    if_then = [
        "If the deadline is larger than expected, stop at ‘draft saved’ and re-run the council.",
        "If energy dips, swap focus for a recovery block and keep only one admin step.",
    ]
    tradeoffs = [
        "Easy to overrun; set a hard stop time before you start.",
        "If supplies are low, don’t ignore them—bundle one essential action.",
    ]
    return {
        "archetype": "admin_first",
        "title": "Admin-first",
        "plan_blocks": blocks,
        "timeline": timeline,
        "timeline_total_min": tmin,
        "timeline_total_max": tmax,
        "wins": wins,
        "top_priorities": top,
        "rationale": rationale,
        "start_here": start_here,
        "stop_rule": stop_rule,
        "if_then": if_then,
        "tradeoffs": tradeoffs,
    }


def merge_llm_proposal(base: dict[str, Any], llm: dict[str, Any] | None) -> dict[str, Any]:
    if not llm:
        return base
    merged = dict(base)
    if isinstance(llm.get("title"), str) and llm["title"].strip():
        merged["title"] = llm["title"].strip()
    for key in ["plan_blocks", "top_priorities", "rationale", "tradeoffs", "start_here", "wins", "if_then"]:
        value = llm.get(key)
        if isinstance(value, list) and any(isinstance(x, str) and x.strip() for x in value):
            merged[key] = [str(x).strip() for x in value if isinstance(x, str) and x.strip()]
    stop_rule = llm.get("stop_rule")
    if isinstance(stop_rule, str) and stop_rule.strip():
        merged["stop_rule"] = stop_rule.strip()
    timeline = llm.get("timeline")
    if isinstance(timeline, list) and timeline:
        merged["timeline"] = timeline
    return merged


def _make_markdown(
    *,
    selected: dict[str, Any],
    energy: EnergyLevel | None,
    deadline_risk: str,
    inventory_risk: str,
    logs: list[dict],
    proposals: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# BabyHandoff — Shift Handoff")
    lines.append("")
    lines.append(f"## Selected Plan: {selected['title']}")
    lines.append("")
    lines.append("### Why this plan won")
    for r in (selected.get("rationale") or [])[:3]:
        lines.append(f"- {r}")
    wins = selected.get("wins") or []
    if isinstance(wins, list) and any(isinstance(x, str) and x.strip() for x in wins):
        lines.append("")
        lines.append("### What you get")
        for w in wins[:5]:
            if isinstance(w, str) and w.strip():
                lines.append(f"- {w.strip()}")
    tradeoffs = (selected.get("tradeoffs") or [])[:3]
    if tradeoffs:
        lines.append("")
        lines.append("### Tradeoffs (what we’re *not* doing)")
        for t in tradeoffs:
            lines.append(f"- {t}")
    lines.append("")

    deadline_items = [l for l in logs if l.get("type") == "deadline" and (l.get("text") or "").strip()]
    inventory_items = [l for l in logs if l.get("type") == "inventory" and (l.get("text") or "").strip()]
    task_items = [l for l in logs if l.get("type") == "task" and (l.get("text") or "").strip()]
    nearest_deadline = str(deadline_items[0].get("text") or "").strip() if deadline_items else ""
    top_inventory = str(inventory_items[0].get("text") or "").strip() if inventory_items else ""
    next_task = str(task_items[0].get("text") or "").strip() if task_items else ""

    lines.append("## Takeover Checklist (start here)")
    start_here = selected.get("start_here") or []
    if isinstance(start_here, list) and any(isinstance(x, str) and x.strip() for x in start_here):
        for s in start_here[:6]:
            if isinstance(s, str) and s.strip():
                lines.append(f"- {s.strip()}")
    else:
        lines.append("- **Scan risks:** confirm what’s actually urgent right now (not what feels loud).")
        if energy == "low" or str(selected.get("archetype")) == "sleep_first":
            lines.append("- **Protect recovery:** set a **60–90 min recovery block** (timer on). If you can’t sleep, do eyes-closed rest.")
    if nearest_deadline:
        lines.append(f"- **Deadline:** {nearest_deadline} (do the first concrete step now).")
    if top_inventory:
        lines.append(f"- **Supplies:** {top_inventory} (one quick stop or add to delivery list).")
    if next_task and next_task not in {nearest_deadline}:
        lines.append(f"- If energy allows: {next_task}")
    stop_rule = selected.get("stop_rule")
    if isinstance(stop_rule, str) and stop_rule.strip():
        lines.append(f"- **Stop rule:** {stop_rule.strip()}")
    lines.append("")

    lines.append("## Next 12h Plan (blocks)")
    for b in selected.get("plan_blocks") or []:
        lines.append(f"- {b}")
    lines.append("")
    lines.append("## Top Priorities")
    for p in (selected.get("top_priorities") or [])[:5]:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## Risks (next 24h)")
    lines.append(f"- Deadline risk: **{deadline_risk}**" + (f" · {nearest_deadline}" if nearest_deadline else ""))
    lines.append(f"- Inventory risk: **{inventory_risk}**" + (f" · {top_inventory}" if top_inventory else ""))
    lines.append(f"- Energy: **{energy or 'unknown'}**")
    lines.append("- If the plan feels wrong in reality: add one note and re-run the council.")
    if_then = selected.get("if_then") or []
    if isinstance(if_then, list) and any(isinstance(x, str) and x.strip() for x in if_then):
        lines.append("")
        lines.append("## If/then (replan cues)")
        for it in if_then[:5]:
            if isinstance(it, str) and it.strip():
                lines.append(f"- {it.strip()}")
    lines.append("")
    lines.append("## Notes (raw)")
    for l in logs[-12:]:
        lines.append(f"- [{str(l.get('type') or '').upper()}] {str(l.get('text') or '').strip()}")
    lines.append("")

    if proposals:
        others = [p for p in proposals if str(p.get("archetype")) != str(selected.get("archetype"))]
        if others:
            lines.append("---")
            lines.append("")
            lines.append("## Other plans considered")
            lines.append("")
            for p in others:
                title = str(p.get("title") or p.get("archetype") or "Plan").strip()
                lines.append(f"### {title}")
                trade = (p.get("tradeoffs") or [])[:2]
                if trade:
                    lines.append("")
                    lines.append("**Tradeoffs**")
                    for t in trade:
                        lines.append(f"- {t}")
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
    source = proposal.get("source") if isinstance(proposal.get("source"), dict) else {}
    provider = str(source.get("provider") or "").strip() if isinstance(source, dict) else ""
    model = str(source.get("model") or "").strip() if isinstance(source, dict) else ""
    if provider or model:
        lines.append(f"_Source: {provider or 'unknown'}" + (f" · {model}_" if model else "_"))
        lines.append("")
    start_here = proposal.get("start_here") or []
    if isinstance(start_here, list) and any(isinstance(x, str) and x.strip() for x in start_here):
        lines.append("**Start here (next 10–15 min)**")
        for s in start_here[:6]:
            if isinstance(s, str) and s.strip():
                lines.append(f"- {s.strip()}")
        lines.append("")
    wins = proposal.get("wins") or []
    if isinstance(wins, list) and any(isinstance(x, str) and x.strip() for x in wins):
        lines.append("**What you get**")
        for w in wins[:6]:
            if isinstance(w, str) and w.strip():
                lines.append(f"- {w.strip()}")
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
    stop_rule = proposal.get("stop_rule")
    if isinstance(stop_rule, str) and stop_rule.strip():
        lines.append("")
        lines.append("**Stop rule**")
        lines.append(f"- {stop_rule.strip()}")
    if_then = proposal.get("if_then") or []
    if isinstance(if_then, list) and any(isinstance(x, str) and x.strip() for x in if_then):
        lines.append("")
        lines.append("**If/then**")
        for it in if_then[:6]:
            if isinstance(it, str) and it.strip():
                lines.append(f"- {it.strip()}")
    tradeoffs = proposal.get("tradeoffs") or []
    if tradeoffs:
        lines.append("")
        lines.append("**Tradeoffs**")
        for t in tradeoffs:
            lines.append(f"- {t}")
    return "\n".join(lines).strip() + "\n"


def decision_to_markdown(decision: dict[str, Any]) -> str:
    selected = str(decision.get("selected_archetype") or "").strip()
    summary = str(decision.get("summary") or "").strip()
    judge_mode = str(decision.get("judge_mode") or "").strip()
    judge_model = str(decision.get("judge_model") or "").strip()
    lines: list[str] = []
    lines.append(f"### Decision: {selected.replace('_', '-')}" if selected else "### Decision")
    lines.append("")
    if judge_mode:
        lines.append(f"_Referee: {judge_mode}" + (f" · {judge_model}_" if judge_model else "_"))
        lines.append("")
    if summary:
        lines.append(summary)
        lines.append("")
    lines.append("**Scores (total / commitments / deadline / energy / time / inventory)**")
    scores = decision.get("scores") or {}
    for archetype, s in scores.items():
        try:
            total = s["_total"]["score"]
            c = s["commitments_fit"]["score"]
            dl = s["deadline_risk"]["score"]
            en = s["energy_match"]["score"]
            tw = s["time_window"]["score"]
            inv = s["inventory_risk"]["score"]
            lines.append(f"- {archetype.replace('_', '-')}: {total} / {c} / {dl} / {en} / {tw} / {inv}")
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


def build_council_transcript_markdown_from_messages(
    *,
    shift_id: str,
    run_id: str,
    messages: list[dict[str, Any]],
    council_mode: str | None = None,
    energy_override: str | None = None,
    context_pack: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# BabyHandoff — Council Transcript")
    lines.append("")
    lines.append("_BabyHandoff is a planning tool. It does not provide medical advice._")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append(f"- Shift: `{shift_id}`")
    lines.append(f"- Run: `{run_id}`")
    if council_mode:
        lines.append(f"- Council mode: `{council_mode}`")
    if energy_override is not None:
        lines.append(f"- Energy override: `{energy_override or 'auto'}`")
    lines.append("")

    if context_pack:
        lines.append("## Context Pack (token-limited)")
        lines.append("")
        summary = str(context_pack.get("summary") or "").strip()
        token_budget = context_pack.get("token_budget")
        token_estimate = context_pack.get("token_estimate")
        if token_budget is not None or token_estimate is not None:
            lines.append(f"- Token estimate / budget: `{token_estimate}` / `{token_budget}`")
            lines.append("")
        if summary:
            lines.append("### Summary")
            lines.append(summary)
            lines.append("")
        compiled = str(context_pack.get("compiled_text") or "").strip()
        if compiled:
            lines.append("### Notes")
            lines.append(compiled)
            lines.append("")

    if events:
        lines.append("## Event Trail")
        lines.append("")
        for e in events[-25:]:
            when = e.get("created_at")
            agent = str(e.get("agent") or "System")
            t = str(e.get("type") or "event")
            msg = str(e.get("message") or "").strip()
            stamp = f"{when}" if when else ""
            if msg:
                lines.append(f"- {stamp} · {agent} · {t}: {msg}")
            else:
                lines.append(f"- {stamp} · {agent} · {t}")
        lines.append("")

    for m in messages:
        sender = str(m.get("sender") or "System")
        created_at = m.get("created_at")
        when = str(created_at) if created_at else ""
        role = str(m.get("role") or "").strip()
        meta = m.get("meta") or {}
        kind = str(meta.get("kind") or "").strip()
        archetype = str(meta.get("archetype") or "").strip()
        provider = str(meta.get("provider") or "").strip()
        model = str(meta.get("model") or "").strip()
        judge_mode = str(meta.get("judge_mode") or "").strip()
        judge_model = str(meta.get("judge_model") or "").strip()
        extras: list[str] = []
        if kind:
            extras.append(kind)
        if archetype:
            extras.append(archetype.replace("_", "-"))
        if provider:
            extras.append(provider + (f"/{model}" if model else ""))
        if judge_mode:
            extras.append(judge_mode + (f"/{judge_model}" if judge_model else ""))
        if meta.get("fallback"):
            extras.append("fallback")
        header = (
            f"## {sender}"
            + (f" · {when}" if when else "")
            + (f" · {role}" if role else "")
            + (f" · {' · '.join(extras)}" if extras else "")
        )
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
        _proposal_sleep_first(energy=energy, deadline_risk=deadline_risk, inventory_risk=inventory_risk, logs=logs),
        _proposal_errands_first(energy=energy, inventory_risk=inventory_risk),
        _proposal_admin_first(energy=energy, deadline_risk=deadline_risk, inventory_risk=inventory_risk),
    ]
    proposals = (
        [merge_llm_proposal(p, proposal_overrides.get(p["archetype"])) for p in base_proposals]
        if proposal_overrides
        else base_proposals
    )

    score_map: dict[str, Any] = {}
    scored: list[tuple[int, int, int, int, dict[str, Any]]] = []
    for p in proposals:
        archetype: Archetype = p["archetype"]
        scores = _score_proposal(archetype, energy=energy, deadline_risk=deadline_risk, inventory_risk=inventory_risk)
        score_map[archetype] = scores
        total = int(scores["_total"]["score"])
        dl = int(scores["deadline_risk"]["score"])
        inv = int(scores["inventory_risk"]["score"])
        en = int(scores["energy_match"]["score"])
        scored.append((total, dl, inv, en, p))

    scored.sort(key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True)
    winner = scored[0][4]

    evidence: list[dict[str, Any]] = []
    evidence.extend(_top_evidence(logs, log_type="deadline"))
    evidence.extend(_top_evidence(logs, log_type="inventory"))
    evidence.extend(_top_evidence(logs, log_type="note"))

    runner_up = scored[1][4] if len(scored) > 1 else None
    reasons: list[str] = []
    if runner_up:
        keys = {
            "deadline_risk": "deadline handling",
            "inventory_risk": "inventory coverage",
            "energy_match": "energy match",
            "time_window": "time realism",
            "commitments_fit": "commitments fit",
        }
        w_scores = score_map.get(winner["archetype"], {})
        r_scores = score_map.get(runner_up["archetype"], {})
        diffs: list[tuple[int, str]] = []
        for k, label in keys.items():
            try:
                diffs.append((int(w_scores[k]["score"]) - int(r_scores[k]["score"]), label))
            except Exception:
                continue
        diffs.sort(key=lambda x: x[0], reverse=True)
        for d, label in diffs:
            if d > 0:
                reasons.append(f"Stronger {label} (+{d} vs next-best).")
            if len(reasons) >= 2:
                break

    decision = {
        "selected_archetype": winner["archetype"],
        "summary": " ".join(
            [
                f"Selected {winner['title']} based on rubric scoring.",
                f"Signals: energy {energy or 'unknown'} · deadline {deadline_risk} · inventory {inventory_risk}.",
                *reasons,
            ]
        ).strip(),
        "scores": score_map,
        "evidence_links": evidence[:6],
    }

    handoff_md = _make_markdown(
        selected=winner,
        energy=energy,
        deadline_risk=deadline_risk,
        inventory_risk=inventory_risk,
        logs=logs,
        proposals=proposals,
    )
    return CouncilOutput(proposals=proposals, decision=decision, handoff_markdown=handoff_md)


def make_handoff_markdown_from_proposals(
    *,
    proposals: list[dict[str, Any]],
    selected_archetype: str,
    logs: list[dict],
    energy_override: str | None,
) -> str:
    """
    Regenerate the handoff markdown for an explicit winner (e.g., when an LLM judge
    overrides the heuristic rubric selection).
    """
    if not proposals:
        return _make_markdown(
            selected={"title": "No plan", "plan_blocks": [], "top_priorities": [], "rationale": [], "archetype": ""},
            energy=_normalize_energy(energy_override, logs),
            deadline_risk=infer_deadline_risk(logs),
            inventory_risk=infer_inventory_risk(logs),
            logs=logs,
        )
    selected = next((p for p in proposals if str(p.get("archetype")) == selected_archetype), proposals[0])
    energy = _normalize_energy(energy_override, logs)
    deadline_risk = infer_deadline_risk(logs)
    inventory_risk = infer_inventory_risk(logs)
    return _make_markdown(
        selected=selected,
        energy=energy,
        deadline_risk=deadline_risk,
        inventory_risk=inventory_risk,
        logs=logs,
        proposals=proposals,
    )
