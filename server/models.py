from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


LogType = Literal["note", "task", "deadline", "inventory"]
EnergyLevel = Literal["high", "medium", "low"]


class Shift(BaseModel):
    id: str = Field(alias="_id")
    created_at: datetime
    updated_at: datetime


class Log(BaseModel):
    id: str = Field(alias="_id")
    shift_id: str
    type: LogType
    text: str
    created_at: datetime


class ShiftState(BaseModel):
    id: str = Field(alias="_id")
    shift_id: str
    energy_inferred: EnergyLevel | None = None
    energy_override: EnergyLevel | None = None
    deadline_risk: Literal["none", "low", "medium", "high"] = "none"
    inventory_risk: Literal["none", "low", "medium", "high"] = "none"
    suggested_next_actions: list[str] = Field(default_factory=list)
    updated_at: datetime


class Proposal(BaseModel):
    id: str = Field(alias="_id")
    run_id: str
    archetype: Literal["sleep_first", "errands_first", "admin_first"]
    title: str
    plan_blocks: list[str]
    top_priorities: list[str]
    rationale: list[str]
    created_at: datetime


class Decision(BaseModel):
    id: str = Field(alias="_id")
    run_id: str
    selected_archetype: Literal["sleep_first", "errands_first", "admin_first"]
    scores: dict[str, dict[str, Any]]
    summary: str
    evidence_links: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class Artifact(BaseModel):
    id: str = Field(alias="_id")
    shift_id: str
    run_id: str
    kind: Literal["handoff_markdown", "council_transcript_markdown"]
    version: int
    markdown: str
    diff_from_artifact_id: str | None = None
    created_at: datetime
