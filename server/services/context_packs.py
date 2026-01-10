from __future__ import annotations

from typing import Any

from server.agents.risk_radar import run_risk_radar


def _logs_to_bullets(logs: list[dict]) -> str:
    lines: list[str] = []
    for l in logs:
        t = str(l.get("type") or "note").upper()
        txt = str(l.get("text") or "").strip()
        if not txt:
            continue
        lines.append(f"- [{t}] {txt}")
    return "\n".join(lines).strip()


def build_context_pack(
    *,
    logs: list[dict],
    energy_override: str | None,
    max_logs: int = 40,
) -> dict[str, Any]:
    trimmed = logs[-max_logs:] if len(logs) > max_logs else list(logs)
    radar = run_risk_radar(trimmed)

    refs: list[dict[str, Any]] = []
    for l in trimmed:
        refs.append(
            {
                "log_id": str(l.get("_id")),
                "type": l.get("type"),
                "text": l.get("text"),
                "created_at": l.get("created_at"),
            }
        )

    summary_lines = [
        f"Energy override: {energy_override or 'auto'}",
        f"Energy inferred: {radar.energy_inferred or 'unknown'}",
        f"Deadline risk: {radar.deadline_risk}",
        f"Inventory risk: {radar.inventory_risk}",
    ]
    for a in radar.suggested_next_actions:
        summary_lines.append(f"Next action: {a}")

    return {
        "energy_override": energy_override,
        "summary": "\n".join(summary_lines).strip(),
        "log_refs": refs,
        "compiled_text": _logs_to_bullets(trimmed),
    }
