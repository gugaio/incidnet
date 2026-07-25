from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncContextManager
from zoneinfo import ZoneInfo

from ..config import DEFAULT_TIMEZONE
from ..log import get_logger
from .base import ConfigField, MCPAdapter, MCPConfigurationError, MCPQueryError
from .registry import register
from .transport import sse_session

logger = get_logger(__name__)


SAFE_NQL_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$")

QUERY_TOOL_NAME = "npaw_query_data"

DEFAULT_LOOKBACK_HOURS = 48

DEFAULT_NPAW_PROMPT = """# Diretrizes de Investigação

## 1. Defina primeiro o escopo do incidente

- Leia o **título e a descrição do incidente** antes de analisar a telemetria.
- Determine qual família e subtipo de device motivou o atendimento.
- Analise somente as sessões compatíveis com esse escopo.
- Um mesmo usuário pode usar vários devices. Ignore completamente sessões de
  outros devices, mesmo que elas apresentem erros.
- Nunca use uma sessão fora do escopo para classificar o usuário.
- Se título e descrição não permitirem identificar o device alvo, declare a
  ambiguidade; não escolha um device por conta própria.
- Se o device alvo estiver claro, mas não houver rows compatíveis, informe
  explicitamente: `nenhum dado encontrado para o device alvo`.
- Ausência de rows compatíveis não confirma falha: classifique como
  `INCONCLUSIVE`, nunca como `BAD`.

## 2. Taxonomia de devices

As famílias principais são `web`, `roku`, `android` e `ios`.

Na resposta NPAW, considere estes nomes equivalentes:

- `player_type` lógico = código NQL `extraparam8` = coluna `Extraparam8`.
- `player_name` lógico = código NQL `player` = coluna `Player`.

### Mobile

Use `player_type` como campo principal:

| player_type | Classificação |
|---|---|
| `ios` | iOS mobile |
| `android` | Android mobile |

Não interprete `player_type=ios` ou `player_type=android` como TV.

### Roku

Use `player_type` como campo principal:

| player_type | Classificação |
|---|---|
| `roku` | Roku |
| `roku_4k` | Roku |
| `roku_4k_hdr` | Roku |

### Web, desktop e TVs

Para desktop, TVs HTML e TVs nativas, use `player_name` como campo principal:

| player_name | Classificação |
|---|---|
| `clappr-web` | Desktop |
| `clappr-web-tvs` | TVs HTML |
| `clappr-native-tvs` | TVs nativas |

- `web` pode significar desktop, TV HTML ou TV nativa. Não assuma que todo
  `web` é desktop.
- Quando título ou descrição informar um fabricante, como LG ou Samsung, use-o
  como filtro adicional do campo `device`/`device_vendor`.
- Android e iOS também podem existir em TVs. Para incidentes explicitamente de
  Android TV ou iOS TV, use os campos contextuais de device/plataforma e a
  descrição do incidente; não reutilize sessões mobile apenas porque
  `player_type` contém `android` ou `ios`.

## 3. Classificação de saúde

- Use somente estes status:
  - `GOOD`: há dados suficientes do device alvo e não há evidência de falha.
  - `BAD`: há evidência de erro, buffering ou comportamento degradado em uma
    ou mais sessões do device alvo.
  - `INCONCLUSIVE`: não foi possível identificar o device alvo, não há rows
    compatíveis ou a consulta não trouxe evidência suficiente.
- A classificação deve considerar somente as rows do device alvo.
- Erro de inicialização ou buffer zerado por mais de 30 segundos indica falha.
- HTTP 404 em fragmentos `.m4s` deve ser isolado como possível problema de CDN.
- Não invente uma causa quando a telemetria não trouxer evidência suficiente.
- A justificativa deve mencionar o device analisado e citar as métricas ou rows
  que sustentam a conclusão.
"""


def _unique_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for raw_name in headers:
        name = raw_name.strip()
        counts[name] = counts.get(name, 0) + 1
        unique.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return unique


def parse_query_result(result: Any) -> list[dict[str, str]]:
    if getattr(result, "isError", False):
        raise MCPQueryError("A tool npaw_query_data retornou erro")
    texts = [
        str(getattr(block, "text"))
        for block in getattr(result, "content", [])
        if getattr(block, "text", None) is not None
    ]
    if not texts:
        raise MCPQueryError("A tool não retornou conteúdo textual")
    try:
        payload = json.loads("\n".join(texts))
    except json.JSONDecodeError as exc:
        raise MCPQueryError("Resposta interna da NPAW não é JSON válido") from exc
    if payload.get("error"):
        raise MCPQueryError(str(payload["error"]))
    if payload.get("status") != "success":
        raise MCPQueryError(f"Status inesperado: {payload.get('status')!r}")
    data = str(payload.get("data", ""))
    lines = [
        line
        for line in data.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return []
    reader = csv.reader(io.StringIO("\n".join(lines)))
    try:
        headers = _unique_headers(next(reader))
    except StopIteration:
        return []
    if not headers or all(not header for header in headers):
        raise MCPQueryError("Resposta sem cabeçalho CSV")
    return [
        dict(zip(headers, row, strict=False))
        for row in reader
        if any(cell.strip() for cell in row)
    ]


def _validated_nql_value(value: str, label: str) -> str:
    if not SAFE_NQL_VALUE.fullmatch(value):
        raise MCPQueryError(f"{label} contém caracteres não permitidos")
    return value


def _nql_start_datetime(value: str) -> str:
    for format_string in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(value, format_string)
            return parsed.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    raise MCPQueryError("Data inicial inválida para NQL")


def build_user_sessions_nql(
    user_id: str,
    period_start: str,
    *,
    player_type_dimension: str = "extraparam8",
    player_name_dimension: str = "player",
) -> str:
    user_id = _validated_nql_value(user_id, "user_id")
    start = _nql_start_datetime(period_start)
    return (
        "select views, playtimeSeconds, errors, bufferRatio "
        f"where datetime >= '{start}' "
        f"and user = '{user_id}' "
        f"group by session_root, {player_type_dimension}, "
        f"{player_name_dimension}, device_type, device"
    )


def build_session_detail_nql(
    user_id: str, session_id: str, period_start: str
) -> str:
    user_id = _validated_nql_value(user_id, "user_id")
    session_id = _validated_nql_value(session_id, "session_root")
    start = _nql_start_datetime(period_start)
    return (
        "select views, playtimeSeconds, errors, stops, healthyPlays, "
        "bufferRatio, bitrate, playsWithError, startupError "
        f"where datetime >= '{start}' "
        f"and user = '{user_id}' "
        f"and session_root = '{session_id}' "
        "group by session_root"
    )


def _decimal(row: dict[str, str], *names: str) -> Decimal:
    lowered = {key.strip().lower(): value for key, value in row.items()}
    for name in names:
        requested = name.lower()
        for header, raw in lowered.items():
            matches = header == requested
            if requested == "views":
                matches = matches or header.startswith("plays (#")
            elif requested == "playtimeseconds":
                matches = matches or header.startswith("avg. playtime")
            elif requested == "errors":
                matches = matches or header.startswith("errors ")
            elif requested == "bufferratio":
                matches = matches or header.startswith("buffer ratio ")
            elif requested == "stops":
                matches = matches or header.startswith("stops ")
            elif requested == "healthyplays":
                matches = matches or header.startswith("healthy plays ")
            elif requested == "bitrate":
                matches = matches or header.startswith("avg. bitrate")
            elif requested == "playswitherror":
                matches = matches or header.startswith("plays with error")
            elif requested == "startuperror":
                matches = matches or header.startswith("startup error")
            if matches and raw not in (None, ""):
                try:
                    return Decimal(str(raw).replace("%", "").strip())
                except InvalidOperation:
                    continue
    return Decimal(0)


def is_suspicious(row: dict[str, str]) -> bool:
    return (
        _decimal(row, "errors") > 0
        or _decimal(row, "bufferRatio") > 0
        or _decimal(row, "playsWithError") > 0
        or _decimal(row, "startupError") > 0
    )


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, int | str]:
    session_ids = {
        value
        for row in rows
        for key, value in row.items()
        if key.lower() == "session_root" and value and value.upper() != "ALL"
    }
    plays = sum((_decimal(row, "views") for row in rows), Decimal(0))
    errors = sum(
        1
        for row in rows
        if _decimal(row, "errors") > 0
        or _decimal(row, "playsWithError") > 0
        or _decimal(row, "startupError") > 0
    )
    buffering = sum(1 for row in rows if _decimal(row, "bufferRatio") > 0)
    return {
        "sessions": len(session_ids) if session_ids else len(rows),
        "plays": str(plays),
        "sessions_with_errors": errors,
        "sessions_with_buffering": buffering,
    }


def deterministic_classification(
    rows: list[dict[str, str]],
) -> tuple[str, str]:
    if not rows:
        return "INCONCLUSIVE", "Nenhum dado encontrado no período consultado."
    suspicious = sum(1 for row in rows if is_suspicious(row))
    if suspicious == 0:
        return "GOOD", "As sessões consultadas não apresentam erros nem buffering."
    if suspicious == len(rows):
        return "BAD", f"Todas as {len(rows)} linhas de telemetria apresentam sinais de falha."
    return (
        "BAD",
        f"{suspicious} de {len(rows)} linhas apresentam erros ou buffering.",
    )


def _normalized(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).lower()


def _row_value(row: dict[str, str], name: str) -> str:
    return next(
        (
            str(value).strip().lower()
            for key, value in row.items()
            if key.lower() == name.lower()
        ),
        "",
    )


def _row_value_any(row: dict[str, str], *names: str) -> str:
    for name in names:
        value = _row_value(row, name)
        if value:
            return value
    return ""


def scope_rows_for_incident(
    rows: list[dict[str, str]], title: str, description: str
) -> tuple[list[dict[str, str]], str]:
    """Filter only when title/description identify a supported device scope."""
    context = _normalized(f"{title} {description}")
    player_types: set[str] = set()
    player_names: set[str] = set()
    target = "não identificado"
    vendor = next(
        (
            candidate
            for candidate, pattern in {
                "lg": r"\blgs?\b",
                "samsung": r"\bsamsung\b",
            }.items()
            if re.search(pattern, context)
        ),
        "",
    )

    def matching_vendor(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
        if not vendor:
            return candidates
        return [
            row
            for row in candidates
            if vendor in _row_value(row, "device")
            or vendor in _row_value(row, "device_vendor")
        ]

    if "roku" in context:
        target = "Roku"
        player_types = {"roku", "roku_4k", "roku_4k_hdr"}
    elif any(term in context for term in ("tv html", "tvs html", "html tv")):
        target = "TV HTML"
        player_names = {"clappr-web-tvs"}
    elif re.search(r"\btvs?\b.*\bnativ", context) or any(
        term in context for term in ("native tv", "native television")
    ):
        target = "TV nativa"
        player_names = {"clappr-native-tvs"}
    elif "desktop" in context:
        target = "Desktop"
        player_names = {"clappr-web"}
    elif "android" in context and "tv" not in context:
        target = "Android mobile"
        player_types = {"android"}
    elif any(term in context for term in (" ios", "ios ", "iphone", "ipad")) and "tv" not in context:
        target = "iOS mobile"
        player_types = {"ios"}
    elif "web" in context and "tv" not in context:
        target = "Web"
        player_names = {"clappr-web", "clappr-web-tvs", "clappr-native-tvs"}
    elif "tv" in context:
        target = "TV"
        scoped = [
            row
            for row in rows
            if _row_value_any(row, "player", "player_name")
            in {"clappr-web-tvs", "clappr-native-tvs"}
            or "tv" in _row_value(row, "device_type")
            or "tv" in _row_value(row, "device")
        ]
        return matching_vendor(scoped), target
    else:
        return rows, target

    scoped = [
        row
        for row in rows
        if (
            player_types
            and _row_value_any(row, "extraparam8", "player_type")
            in player_types
        )
        or (
            player_names
            and _row_value_any(row, "player", "player_name")
            in player_names
        )
    ]
    return matching_vendor(scoped), target


class NpawSettings:
    """Plain container for NPAW-specific settings, validated on construction."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.url = str(raw.get("url", "")).strip()
        self.api_key = str(raw.get("api_key", "")).strip()
        self.account_code = str(raw.get("account_code", "")).strip()
        self.environment = str(raw.get("environment", "") or "prod").strip()
        self.timezone = str(raw.get("timezone", "") or DEFAULT_TIMEZONE).strip()
        try:
            self.lookback_hours = int(raw.get("lookback_hours") or DEFAULT_LOOKBACK_HOURS)
        except (TypeError, ValueError):
            self.lookback_hours = DEFAULT_LOOKBACK_HOURS
        self.player_type_dimension = str(
            raw.get("player_type_dimension", "") or "extraparam8"
        ).strip()
        self.player_name_dimension = str(
            raw.get("player_name_dimension", "") or "player"
        ).strip()


@register
class NpawAdapter(MCPAdapter):
    """MCP adapter for the NPAW streaming analytics platform (NQL over SSE)."""

    provider_id = "npaw"
    label = "NPAW (NQL)"

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        super().__init__(settings)
        self.settings = NpawSettings(self.raw_settings)

    @classmethod
    def config_fields(cls) -> list[ConfigField]:
        return [
            ConfigField(
                name="url",
                label="URL do MCP (SSE)",
                type="text",
                help="Endpoint SSE do servidor MCP da NPAW.",
            ),
            ConfigField(
                name="api_key",
                label="API key",
                type="password",
                secret=True,
            ),
            ConfigField(
                name="account_code",
                label="Account code",
                type="text",
            ),
            ConfigField(
                name="environment",
                label="Environment",
                type="text",
                default="prod",
            ),
            ConfigField(
                name="lookback_hours",
                label="Janela de investigação (horas)",
                type="number",
                default=str(DEFAULT_LOOKBACK_HOURS),
            ),
            ConfigField(
                name="timezone",
                label="Timezone",
                type="text",
                default=DEFAULT_TIMEZONE,
            ),
        ]

    def validate_config(self) -> bool:
        return bool(
            self.settings.url and self.settings.api_key and self.settings.account_code
        )

    def required_tools(self) -> set[str]:
        return {QUERY_TOOL_NAME}

    def secret(self) -> str:
        return self.settings.api_key

    def default_prompt(self) -> str:
        return DEFAULT_NPAW_PROMPT

    def classify(self, rows: list[dict[str, str]]) -> tuple[str, str]:
        return deterministic_classification(rows)

    def _headers(self) -> dict[str, str]:
        if not self.settings.url:
            raise MCPConfigurationError("URL do MCP NPAW não configurada")
        if not self.settings.api_key:
            raise MCPConfigurationError("API key da NPAW não configurada")
        if not self.settings.account_code:
            raise MCPConfigurationError("Account code da NPAW não configurado")
        return {
            "npaw-api-key": self.settings.api_key,
            "npaw-account-code": self.settings.account_code,
            "npaw-environment": self.settings.environment or "prod",
        }

    def session(self) -> AsyncContextManager[Any]:
        headers = self._headers()
        logger.debug(
            "Conectando ao MCP NPAW",
            extra={"url": self.settings.url, "environment": self.settings.environment},
        )
        return sse_session(
            self.settings.url, headers, timeout=30, sse_read_timeout=120
        )

    async def query_user(
        self,
        session: Any,
        user_id: str,
        *,
        incident_title: str = "",
        incident_description: str = "",
    ) -> dict[str, Any]:
        now = datetime.now(ZoneInfo(self.settings.timezone))
        period_start = now - timedelta(hours=self.settings.lookback_hours)
        start_text = period_start.strftime("%Y-%m-%d %H:%M:%S")
        initial_nql = build_user_sessions_nql(
            user_id,
            start_text,
            player_type_dimension=self.settings.player_type_dimension,
            player_name_dimension=self.settings.player_name_dimension,
        )
        logger.debug(
            "Consultando sessões do usuário",
            extra={"user_id": user_id, "lookback_hours": self.settings.lookback_hours},
        )
        result = await session.call_tool(
            QUERY_TOOL_NAME, arguments={"nql": initial_nql, "timeout": 60}
        )
        raw_rows = parse_query_result(result)
        rows, target_device = scope_rows_for_incident(
            raw_rows, incident_title, incident_description
        )
        queries = [initial_nql]
        detailed_rows: list[dict[str, str]] = []
        suspect_sessions: list[str] = []
        for row in rows:
            if not is_suspicious(row):
                continue
            session_id = next(
                (
                    value
                    for key, value in row.items()
                    if key.lower() == "session_root"
                    and value
                    and value.upper() != "ALL"
                ),
                "",
            )
            if session_id and session_id not in suspect_sessions:
                suspect_sessions.append(session_id)
        logger.debug(
            "Sessões do usuário analisadas",
            extra={
                "user_id": user_id,
                "target_device": target_device,
                "total_rows": len(raw_rows),
                "matched_rows": len(rows),
                "suspect_sessions": len(suspect_sessions),
            },
        )
        for session_id in suspect_sessions[:3]:
            try:
                nql = build_session_detail_nql(user_id, session_id, start_text)
            except MCPQueryError:
                continue
            detail = await session.call_tool(
                QUERY_TOOL_NAME, arguments={"nql": nql, "timeout": 60}
            )
            detailed_rows.extend(parse_query_result(detail))
            queries.append(nql)
        return {
            "period": {
                "from": start_text,
                "to": now.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": self.settings.timezone,
            },
            "scope": {
                "target_device": target_device,
                "matched_rows": len(rows),
                "total_rows": len(raw_rows),
            },
            "queries": queries,
            "rows": rows + detailed_rows,
            "summary": summarize_rows(rows),
        }
