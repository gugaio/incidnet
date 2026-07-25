from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncContextManager, ClassVar

from pydantic import BaseModel

from ..log import get_logger

logger = get_logger(__name__)


class MCPConfigurationError(RuntimeError):
    """Raised when an MCP provider is missing required configuration."""


class MCPQueryError(RuntimeError):
    """Raised when an MCP tool call fails or returns an unexpected shape."""


def redact_exception(exc: BaseException, secrets: tuple[str, ...] = ()) -> str:
    """Render an exception as text, stripping any known secret values."""
    if isinstance(exc, BaseExceptionGroup):
        message = "; ".join(redact_exception(item, secrets) for item in exc.exceptions)
    else:
        message = f"{type(exc).__name__}: {exc}"
    for secret in secrets:
        if secret:
            message = message.replace(secret, "***")
    return message


class ConfigField(BaseModel):
    """Describes one provider-specific setting, for use by the settings UI."""

    name: str
    label: str
    type: str = "text"  # "text" | "password" | "number"
    default: str = ""
    secret: bool = False
    help: str = ""


class MCPAdapter(ABC):
    """Base class every MCP provider integration must implement.

    An adapter owns everything that is specific to one MCP server: how to
    connect to it (transport + auth), which tool(s) it must expose, how to
    turn a request for a user's telemetry into tool calls, and how to reduce
    the raw rows into a deterministic GOOD/BAD/INCONCLUSIVE classification.

    Every adapter normalizes its output into the same telemetry shape so the
    rest of the app (investigator, llm, storage) never has to know which
    provider is in use:

        {
            "period": {"from": str, "to": str, "timezone": str},
            "scope": {"target_device": str, "matched_rows": int, "total_rows": int},
            "queries": list[str],
            "rows": list[dict[str, str]],
            "summary": dict[str, int | str],
        }
    """

    provider_id: ClassVar[str]
    label: ClassVar[str]

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.raw_settings = dict(settings or {})

    @classmethod
    def config_fields(cls) -> list[ConfigField]:
        """Settings this provider needs, described for the settings UI."""
        return []

    @abstractmethod
    def validate_config(self) -> bool:
        """Whether enough settings are present to attempt a connection."""

    @abstractmethod
    def required_tools(self) -> set[str]:
        """MCP tool names this adapter needs the server to expose."""

    @abstractmethod
    def session(self) -> AsyncContextManager[Any]:
        """Async context manager yielding a connected MCP ClientSession."""

    @abstractmethod
    async def query_user(
        self,
        session: Any,
        user_id: str,
        *,
        incident_title: str = "",
        incident_description: str = "",
    ) -> dict[str, Any]:
        """Fetch and normalize telemetry for one user, scoped to the incident."""

    @abstractmethod
    def classify(self, rows: list[dict[str, str]]) -> tuple[str, str]:
        """Deterministic (non-LLM) classification for a set of telemetry rows."""

    def default_prompt(self) -> str:
        """Seed content for a new workspace's PROMPT_BASE.md."""
        return ""

    def secret(self) -> str:
        """Sensitive value(s) that must be redacted from user-facing errors."""
        return ""
