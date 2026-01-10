from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal


EnergyLevel = Literal["high", "medium", "low"]
RiskLevel = Literal["none", "low", "medium", "high"]


@dataclass(frozen=True)
class RiskRadarResult:
    energy_inferred: EnergyLevel | None
    deadline_risk: RiskLevel
    inventory_risk: RiskLevel
    suggested_next_actions: list[str]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def infer_energy(texts: list[str]) -> EnergyLevel | None:
    joined = " ".join(t.lower() for t in texts)
    low_markers = ["low energy", "exhaust", "no sleep", "broken sleep", "3–4 hours", "3-4 hours", "up all night"]
    high_markers = ["high energy", "slept great", "well rested"]
    if any(m in joined for m in low_markers):
        return "low"
    if any(m in joined for m in high_markers):
        return "high"
    return None


def infer_inventory_risk(logs: list[dict]) -> RiskLevel:
    inventory_logs = [l for l in logs if l.get("type") == "inventory"]
    if not inventory_logs:
        return "none"
    joined = " ".join((l.get("text") or "").lower() for l in inventory_logs)
    if any(k in joined for k in ["< 1 day", "out", "almost out", "empty"]):
        return "high"
    if any(k in joined for k in ["low", "running low", "soon"]):
        return "medium"
    return "low"


def infer_deadline_risk(logs: list[dict]) -> RiskLevel:
    deadline_logs = [l for l in logs if l.get("type") == "deadline"]
    if not deadline_logs:
        return "none"
    # MVP: heuristics only; later can parse real datetimes.
    joined = " ".join((l.get("text") or "").lower() for l in deadline_logs)
    if any(k in joined for k in ["today", "tonight", "in 2 hours", "asap"]):
        return "high"
    if any(k in joined for k in ["tomorrow", "by tomorrow", "next day"]):
        return "medium"
    return "low"


def suggest_next_actions(deadline_risk: RiskLevel, inventory_risk: RiskLevel, energy: EnergyLevel | None) -> list[str]:
    actions: list[str] = []
    if deadline_risk in {"high", "medium"}:
        actions.append("Create a 15–30 min admin block to clear the nearest deadline.")
    if inventory_risk in {"high", "medium"}:
        actions.append("Create a single supply run task (diapers/wipes/cream) and keep it to one stop.")
    if energy == "low":
        actions.append("Add one protected rest block and reduce the plan to essentials.")
    return actions[:3]


def run_risk_radar(logs: list[dict]) -> RiskRadarResult:
    texts = [str(l.get("text") or "") for l in logs]
    energy = infer_energy(texts)
    inv = infer_inventory_risk(logs)
    dl = infer_deadline_risk(logs)
    return RiskRadarResult(
        energy_inferred=energy,
        deadline_risk=dl,
        inventory_risk=inv,
        suggested_next_actions=suggest_next_actions(dl, inv, energy),
    )

