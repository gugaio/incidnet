from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from .log import get_logger


logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

WORKSPACE_ROOT = (PROJECT_ROOT / "workspace").resolve()
TEMPLATES_ROOT = PROJECT_ROOT / "templates"

DEFAULT_CRON = "0 0 * * *"
DEFAULT_TIMEZONE = os.getenv("INCIDNET_TIMEZONE", "America/Sao_Paulo")
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip()

# How many users an investigation round analyzes concurrently. Bounded so we
# don't overwhelm the MCP server / LLM provider while still cutting the total
# time for incidents with hundreds/thousands of users.
try:
    INVESTIGATION_CONCURRENCY = max(
        1, int(os.getenv("INCIDNET_INVESTIGATION_CONCURRENCY", "8"))
    )
except ValueError:
    INVESTIGATION_CONCURRENCY = 8
    logger.warning(
        "INCIDNET_INVESTIGATION_CONCURRENCY inválido, usando padrão 8"
    )

logger.info(
    "Configuração carregada",
    extra={
        "timezone": DEFAULT_TIMEZONE,
        "openai_model": DEFAULT_OPENAI_MODEL or "(não configurado)",
        "concurrency": INVESTIGATION_CONCURRENCY,
    },
)
