from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal
import re


EnergyLevel = Literal["high", "medium", "low"]
RiskLevel = Literal["none", "low", "medium", "high"]


@dataclass(frozen=True)
class RiskRadarResult:
    energy_inferred: EnergyLevel | None
    deadline_risk: RiskLevel
    inventory_risk: RiskLevel
    focus_pressure: RiskLevel
    suggested_next_actions: list[str]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def infer_energy(texts: list[str]) -> EnergyLevel | None:
    joined = " ".join(t.lower() for t in texts)
    if "energy: low" in joined or "low energy" in joined:
        return "low"
    if "energy: high" in joined or "high energy" in joined:
        return "high"
    if "energy: medium" in joined or "medium energy" in joined:
        return "medium"

    # Heuristic: parse “slept X hours” and map to energy.
    # Examples: "slept 4h", "slept ~3–4 hours", "sleep 6.5 hours".
    hours: list[float] = []
    for m in re.finditer(r"(?:slept|sleep)\s*~?\s*(\d+(?:\.\d+)?)\s*h(?:ours?)?", joined):
        try:
            hours.append(float(m.group(1)))
        except Exception:
            pass
    for m in re.finditer(r"(\d+)\s*(?:–|-)\s*(\d+)\s*h(?:ours?)?", joined):
        try:
            a = float(m.group(1))
            b = float(m.group(2))
            if 0 < a < 16 and 0 < b < 16:
                hours.append((a + b) / 2.0)
        except Exception:
            pass
    if hours:
        h = max(0.0, min(16.0, sum(hours) / len(hours)))
        if h < 5.0:
            return "low"
        if h >= 7.0:
            return "high"
        return "medium"

    low_markers = ["exhaust", "no sleep", "broken sleep", "up all night", "wiped", "wrecked"]
    high_markers = ["slept great", "well rested", "fresh"]
    if any(m in joined for m in low_markers):
        return "low"
    if any(m in joined for m in high_markers):
        return "high"
    return None


def infer_focus_pressure(logs: list[dict]) -> RiskLevel:
    """
    Focus pressure is about “I need a protected deep-work window soon”
    (school/work admin), independent of strict deadlines.
    """
    joined = " ".join(str(l.get("text") or "").lower() for l in logs)
    if not joined.strip():
        return "none"
    keywords = ["focus block", "deep work", "assignment", "paper", "study", "homework", "deliverable", "exam", "project"]
    if not any(k in joined for k in keywords):
        return "none"

    # If the user already wrote timebox hints, treat it as higher pressure.
    if any(k in joined for k in ["due today", "today", "tonight", "tomorrow", "by tomorrow", "submit"]):
        return "high"
    if any(k in joined for k in ["due in 2 days", "due in 3 days", "this week", "in 2 days", "in 3 days"]):
        return "medium"
    return "low"


def _shorten(s: str, max_len: int = 72) -> str:
    txt = " ".join((s or "").strip().split())
    if len(txt) <= max_len:
        return txt
    return txt[: max_len - 1].rstrip() + "…"


def _first_text(logs: list[dict], *, log_type: str) -> str:
    for l in logs:
        if (l.get("type") == log_type) or (log_type in (l.get("tags") or [])):
            txt = str(l.get("text") or "").strip()
            if txt:
                return _shorten(txt)
    return ""


def infer_inventory_risk(logs: list[dict]) -> RiskLevel:
    inventory_logs = [l for l in logs if (l.get("type") == "inventory") or ("inventory" in (l.get("tags") or []))]
    if not inventory_logs:
        return "none"
    joined = " ".join((l.get("text") or "").lower() for l in inventory_logs)
    if any(k in joined for k in ["< 1 day", "out", "almost out", "empty"]):
        return "high"
    if any(k in joined for k in ["low", "running low", "soon"]):
        return "medium"
    return "low"


def infer_deadline_risk(logs: list[dict]) -> RiskLevel:
    deadline_logs = [l for l in logs if (l.get("type") == "deadline") or ("deadline" in (l.get("tags") or []))]
    if not deadline_logs:
        return "none"
    # MVP: heuristics only; later can parse real datetimes.
    joined = " ".join((l.get("text") or "").lower() for l in deadline_logs)
    if any(k in joined for k in ["today", "tonight", "in 2 hours", "asap"]):
        return "high"
    if any(k in joined for k in ["tomorrow", "by tomorrow", "next day"]):
        return "medium"
    return "low"


def suggest_next_actions(
    logs: list[dict],
    *,
    deadline_risk: RiskLevel,
    inventory_risk: RiskLevel,
    focus_pressure: RiskLevel,
    energy: EnergyLevel | None,
) -> list[str]:
    actions: list[str] = []
    if deadline_risk in {"high", "medium"}:
        dl = _first_text(logs, log_type="deadline")
        actions.append(f"Start the nearest deadline now: “{dl or 'open the form/email'}” (10–20 min, save draft).")
    if inventory_risk in {"high", "medium"}:
        inv = _first_text(logs, log_type="inventory")
        actions.append(f"Cover supplies in one step: “{inv or 'diapers/wipes/cream'}” (pickup/delivery/one stop).")
    if focus_pressure in {"high", "medium"}:
        actions.append("Protect one focus window (30–60 min): timer + single task (no multitasking).")
    if energy == "low":
        actions.append("Protect one recovery block (60–90 min) before adding anything else.")
    return actions[:4]


def run_risk_radar(logs: list[dict]) -> RiskRadarResult:
    texts = [str(l.get("text") or "") for l in logs]
    energy = infer_energy(texts)
    inv = infer_inventory_risk(logs)
    dl = infer_deadline_risk(logs)
    focus = infer_focus_pressure(logs)
    return RiskRadarResult(
        energy_inferred=energy,
        deadline_risk=dl,
        inventory_risk=inv,
        focus_pressure=focus,
        suggested_next_actions=suggest_next_actions(
            logs,
            deadline_risk=dl,
            inventory_risk=inv,
            focus_pressure=focus,
            energy=energy,
        ),
    )
