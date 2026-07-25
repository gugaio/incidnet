from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class MCPConfig(BaseModel):
    """Which MCP provider is in use, and its provider-specific settings.

    `settings` is validated and interpreted by the adapter registered under
    `provider` (see app/adapters). This model intentionally stays generic so
    any MCP provider can be plugged in without changing this schema. The
    application builds a single, process-wide `MCPConfig` from environment
    variables (see `app.adapters.registry.load_env_mcp_config`); it is not
    stored per workspace or editable through the UI.
    """

    provider: str = "npaw"
    settings: dict[str, Any] = Field(default_factory=dict)


class Workspace(BaseModel):
    workspace_id: str
    name: str
    created_at: datetime
    cron_schedule: str = "0 0 * * *"


class Incident(BaseModel):
    incident_id: str
    title: str
    description: str = ""
    status: Literal["OPEN", "RESOLVED", "CLOSED"] = "OPEN"
    created_at: datetime
    affected_users: list[str] = Field(default_factory=list)


class FeedEntry(BaseModel):
    id: str
    timestamp: datetime
    author: str
    author_type: Literal["AGENT", "HUMAN"]
    content: str
    kind: str = "COMMENT"


class Feed(BaseModel):
    entries: list[FeedEntry] = Field(default_factory=list)


class MemoryEntry(BaseModel):
    """A single piece of durable knowledge learned from human feedback.

    Lives either in an incident's memory (learned from that incident's feed) or,
    once consolidated, in the workspace memory (applicable to future incidents).
    """

    id: str
    timestamp: datetime
    content: str
    author: str = "Agente Incidnet"
    source_incident_id: str | None = None


class Memory(BaseModel):
    entries: list[MemoryEntry] = Field(default_factory=list)


class UserDiagnosis(BaseModel):
    user_id: str
    classification: Literal["GOOD", "BAD", "INCONCLUSIVE"]
    justification: str
    investigated_at: datetime
    source: str
    period: dict[str, str | None] = Field(default_factory=dict)
    queries: list[str] = Field(default_factory=list)
    rows: list[dict[str, str]] = Field(default_factory=list)
    summary: dict[str, int | str] = Field(default_factory=dict)
    scope: dict[str, int | str] = Field(default_factory=dict)
    resolved: bool = False
    resolved_at: datetime | None = None
    resolved_by: str | None = None

    @property
    def status(self) -> str:
        return "RESOLVED" if self.resolved else self.classification

    @field_validator("classification", mode="before")
    @classmethod
    def migrate_legacy_classification(cls, value: str) -> str:
        return {
            "HEALTHY": "GOOD",
            "UNHEALTHY": "BAD",
            "INTERMITTENT": "INCONCLUSIVE",
            "ERROR": "INCONCLUSIVE",
        }.get(value, value)


class AnalysisSummary(BaseModel):
    total_users: int = 0
    good: int = 0
    bad: int = 0
    inconclusive: int = 0


class Analysis(BaseModel):
    """Immutable result produced by an investigator or a specialised agent.

    Diagnoses in ``analyses/{analysis_id}/users`` are a snapshot. The mutable
    ``users`` directory remains the operational projection used to resolve or
    reopen a user.
    """

    analysis_id: str
    incident_id: str
    type: str = "telemetry_diagnosis"
    status: Literal["COMPLETED"] = "COMPLETED"
    created_at: datetime
    agent_name: str = "Agente Incidnet"
    parent_analysis_id: str | None = None
    content: str = ""
    summary: AnalysisSummary = Field(default_factory=AnalysisSummary)
