from __future__ import annotations

from .base import ConfigField, MCPAdapter, MCPConfigurationError, MCPQueryError, redact_exception
from .registry import (
    available_providers,
    get_adapter,
    get_adapter_class,
    load_env_mcp_config,
    register,
)

# Import built-in adapters so they self-register on package import.
from . import npaw as _npaw  # noqa: F401

__all__ = [
    "ConfigField",
    "MCPAdapter",
    "MCPConfigurationError",
    "MCPQueryError",
    "redact_exception",
    "available_providers",
    "get_adapter",
    "get_adapter_class",
    "load_env_mcp_config",
    "register",
]
