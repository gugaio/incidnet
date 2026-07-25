from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging import LogRecord


_initialized = False


def setup_logging(*, force: bool = False) -> None:
    global _initialized
    if _initialized and not force:
        return
    _initialized = True

    level_name = os.getenv("INCIDNET_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


class _JsonFormatter(logging.Formatter):
    def format(self, record: LogRecord) -> str:
        result = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "func": record.funcName,
        }
        if record.exc_info and record.exc_info[0] is not None:
            result["exception"] = self.formatException(record.exc_info)
        return json.dumps(result, ensure_ascii=False, default=str)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
