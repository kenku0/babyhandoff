from __future__ import annotations

from typing import Any

from server.agents.risk_radar import run_risk_radar


def _estimate_tokens(text: str) -> int:
    # Rough heuristic (~4 chars/token for English); good enough for budgeting.
    return max(1, (len(text) + 3) // 4)


def _logs_to_bullets(logs: list[dict]) -> str:
    lines: list[str] = []
    for l in logs:
        t = str(l.get("type") or "note").upper()
        txt = str(l.get("text") or "").strip()
        if not txt:
            continue
        lines.append(f"- [{t}] {txt}")
    return "\n".join(lines).strip()


def _pick_logs_for_budget(
    logs: list[dict],
    *,
    token_budget: int,
    energy_override: str | None,
    must_keep_per_type: int = 2,
) -> list[dict]:
    if not logs:
        return []

    # Keep a small "must-keep" set for high-signal categories, then fill with recency.
    by_type: dict[str, list[dict]] = {"deadline": [], "inventory": [], "task": [], "note": []}
    for l in logs:
        t = str(l.get("type") or "note")
        if t not in by_type:
            t = "note"
        by_type[t].append(l)

    # Most-recent first within each type.
    for t in by_type:
        by_type[t] = list(reversed(by_type[t]))

    selected_ids: set[str] = set()
    selected: list[dict] = []

    def try_add(log: dict) -> bool:
        log_id = str(log.get("_id") or "")
        if not log_id or log_id in selected_ids:
            return False
        candidate = selected + [log]
        compiled = _logs_to_bullets(list(reversed(candidate)))  # temporary: newest-first for estimate
        summary_stub = "\n".join(
            [
                f"Energy override: {energy_override or 'auto'}",
                f"Included logs: {len(candidate)}",
            ]
        )
        estimate = _estimate_tokens(summary_stub + "\n\n" + compiled)
        if estimate > token_budget:
            return False
        selected_ids.add(log_id)
        selected.append(log)
        return True

    # Must-keep: deadline/inventory/tasks (most recent first).
    for t in ["deadline", "inventory", "task"]:
        for l in by_type[t][:must_keep_per_type]:
            try_add(l)

    # Fill with most recent logs overall.
    for l in reversed(logs):
        if not try_add(l):
            continue

    # Return chronological order for readability.
    selected_map = {str(l.get("_id")): l for l in selected}
    ordered: list[dict] = []
    for l in logs:
        lid = str(l.get("_id") or "")
        if lid in selected_map:
            ordered.append(l)
    return ordered


def build_context_pack(
    *,
    logs: list[dict],
    energy_override: str | None,
    token_budget: int = 4000,
) -> dict[str, Any]:
    picked = _pick_logs_for_budget(logs, token_budget=token_budget, energy_override=energy_override)
    radar = run_risk_radar(picked)

    refs: list[dict[str, Any]] = []
    for l in picked:
        refs.append(
            {
                "log_id": str(l.get("_id")),
                "type": l.get("type"),
                "text": l.get("text"),
                "created_at": l.get("created_at"),
            }
        )

    normalized: dict[str, Any] = {"deadlines": [], "inventory": [], "tasks": [], "notes": []}
    for r in refs:
        t = str(r.get("type") or "note")
        if t == "deadline":
            normalized["deadlines"].append(r)
        elif t == "inventory":
            normalized["inventory"].append(r)
        elif t == "task":
            normalized["tasks"].append(r)
        else:
            normalized["notes"].append(r)

    summary_lines = [
        f"Energy override: {energy_override or 'auto'}",
        f"Energy inferred: {radar.energy_inferred or 'unknown'}",
        f"Focus pressure: {radar.focus_pressure}",
        f"Deadline risk: {radar.deadline_risk}",
        f"Inventory risk: {radar.inventory_risk}",
        f"Logs included: {len(picked)}/{len(logs)}",
    ]
    for a in radar.suggested_next_actions:
        summary_lines.append(f"Next action: {a}")

    compiled_text = _logs_to_bullets(picked)
    token_estimate = _estimate_tokens("\n".join(summary_lines) + "\n\n" + compiled_text)
    return {
        "energy_override": energy_override,
        "radar": {
            "energy_inferred": radar.energy_inferred,
            "deadline_risk": radar.deadline_risk,
            "inventory_risk": radar.inventory_risk,
            "focus_pressure": radar.focus_pressure,
            "suggested_next_actions": list(radar.suggested_next_actions or []),
        },
        "counts": {"logs_included": len(picked), "logs_total": len(logs)},
        "summary": "\n".join(summary_lines).strip(),
        "log_refs": refs,
        "compiled_text": compiled_text,
        "token_estimate": token_estimate,
        "included_log_ids": [str(l.get("_id") or "") for l in picked if str(l.get("_id") or "")],
        "normalized": normalized,
        "trim_strategy": {
            "must_keep_per_type": 2,
            "priority_types": ["deadline", "inventory", "task", "note"],
            "fill": "recency",
            "estimate": "chars_per_token~4",
        },
    }
