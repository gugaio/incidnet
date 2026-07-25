from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from ..log import get_logger
from .base import MCPConfigurationError

logger = get_logger(__name__)


@asynccontextmanager
async def sse_session(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = 30,
    sse_read_timeout: float = 120,
) -> AsyncIterator[Any]:
    """Connect to an MCP server over Server-Sent Events and initialize it."""
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError as exc:
        raise MCPConfigurationError("Dependência mcp não instalada") from exc
    logger.debug("Iniciando conexão SSE com MCP", extra={"url": url})
    async with sse_client(
        url, headers=headers, timeout=timeout, sse_read_timeout=sse_read_timeout
    ) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await session.send_ping()
            logger.debug("Conexão SSE estabelecida")
            yield session


@asynccontextmanager
async def streamable_http_session(
    url: str,
    headers: dict[str, str],
    *,
    timeout: float = 30,
) -> AsyncIterator[Any]:
    """Connect to an MCP server over streamable HTTP and initialize it."""
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:
        raise MCPConfigurationError("Dependência mcp não instalada") from exc
    async with streamablehttp_client(url, headers=headers, timeout=timeout) as (
        read_stream,
        write_stream,
        _get_session_id,
    ):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield session
