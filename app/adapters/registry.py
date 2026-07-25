from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..log import get_logger
from .base import MCPAdapter, MCPConfigurationError

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..models import MCPConfig

_REGISTRY: dict[str, type[MCPAdapter]] = {}


def register(adapter_cls: type[MCPAdapter]) -> type[MCPAdapter]:
    """Register an adapter class under its `provider_id`. Usable as a decorator."""
    _REGISTRY[adapter_cls.provider_id] = adapter_cls
    return adapter_cls


def available_providers() -> list[type[MCPAdapter]]:
    return list(_REGISTRY.values())


def get_adapter_class(provider_id: str) -> type[MCPAdapter] | None:
    return _REGISTRY.get(provider_id)


def get_adapter(mcp_config: "MCPConfig") -> MCPAdapter:
    adapter_cls = _REGISTRY.get(mcp_config.provider)
    if adapter_cls is None:
        raise MCPConfigurationError(
            f"Provider MCP desconhecido: {mcp_config.provider!r}"
        )
    return adapter_cls(mcp_config.settings)


def load_env_mcp_config() -> "MCPConfig":
    """Build the single, process-wide MCP configuration from environment variables.

    MCP connection details are intentionally not configurable through the UI
    or stored per workspace: operators set them in the project's `.env` (or in
    real environment variables) and restart the app to apply changes.
    """
    from ..models import MCPConfig

    return MCPConfig(
        provider="npaw",
        settings={
            "url": os.getenv("NPAW_MCP_URL", "").strip(),
            "api_key": os.getenv("NPAW_API_KEY", "").strip(),
            "account_code": os.getenv("NPAW_ACCOUNT_CODE", "").strip(),
        },
    )

