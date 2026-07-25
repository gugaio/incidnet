from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.adapters.base import MCPConfigurationError, MCPQueryError, redact_exception
from app.adapters.npaw import (
    NpawAdapter,
    build_user_sessions_nql,
    deterministic_classification,
    parse_query_result,
    scope_rows_for_incident,
    summarize_rows,
)


def result_with(payload: dict, *, is_error: bool = False):
    return SimpleNamespace(
        isError=is_error,
        content=[SimpleNamespace(text=json.dumps(payload))],
    )


def test_parse_query_result_preserves_duplicate_headers_and_csv_quoting():
    result = result_with(
        {
            "status": "success",
            "data": (
                "# Analytics data\n"
                'session_root,Type,Type,device\n'
                'abc,VOD,Smart TV,"Samsung, 2026"\n'
            ),
        }
    )
    assert parse_query_result(result) == [
        {
            "session_root": "abc",
            "Type": "VOD",
            "Type_2": "Smart TV",
            "device": "Samsung, 2026",
        }
    ]


def test_query_error_is_not_treated_as_empty_data():
    with pytest.raises(MCPQueryError):
        parse_query_result(result_with({"error": "HTTP 400: invalid NQL"}))


def test_query_builder_rejects_nql_injection():
    with pytest.raises(MCPQueryError):
        build_user_sessions_nql("x' or user='other", "2026-07-24")


def test_query_builder_uses_validated_player_dimension_codes():
    nql = build_user_sessions_nql("user-101", "2026-07-24")
    assert (
        "group by session_root, extraparam8, player, device_type, device"
        in nql
    )
    assert "player_type" not in nql
    assert "player_name" not in nql


def test_query_builder_accepts_custom_player_dimension_codes():
    nql = build_user_sessions_nql(
        "user-101",
        "2026-07-24",
        player_type_dimension="custom_type",
        player_name_dimension="custom_name",
    )
    assert "group by session_root, custom_type, custom_name, device_type, device" in nql


def test_query_builder_accepts_precise_lookback_timestamp():
    nql = build_user_sessions_nql("user-101", "2026-07-22 20:15:30")
    assert "datetime >= '2026-07-22 20:15:30'" in nql


def test_summary_and_deterministic_classification():
    rows = [
        {
            "session_root": "a",
            "views": "2",
            "errors": "0",
            "bufferRatio": "0",
        },
        {
            "session_root": "b",
            "views": "1",
            "errors": "1",
            "bufferRatio": "2.5",
        },
    ]
    assert summarize_rows(rows) == {
        "sessions": 2,
        "plays": "3",
        "sessions_with_errors": 1,
        "sessions_with_buffering": 1,
    }
    assert deterministic_classification(rows)[0] == "BAD"


def test_summary_reads_human_readable_npaw_headers():
    rows = [
        {
            "Session_root": "session-a",
            "Plays (#) (Plays)": "2",
            "Errors (#) (Errors)": "1",
            "Buffer Ratio (%) (%)": "3.5",
        }
    ]
    assert summarize_rows(rows) == {
        "sessions": 1,
        "plays": "2",
        "sessions_with_errors": 1,
        "sessions_with_buffering": 1,
    }


def test_adapter_validate_config_requires_url_key_and_account_code():
    assert NpawAdapter({}).validate_config() is False
    assert (
        NpawAdapter(
            {
                "url": "https://mcp.example.invalid/sse",
                "api_key": "runtime-value",
                "account_code": "incidnet",
            }
        ).validate_config()
        is True
    )


def test_adapter_headers_use_configured_settings_and_force_default_environment():
    adapter = NpawAdapter(
        {
            "url": "https://mcp.example.invalid/sse",
            "api_key": "runtime-value",
            "account_code": "incidnet",
        }
    )
    assert adapter._headers() == {
        "npaw-api-key": "runtime-value",
        "npaw-account-code": "incidnet",
        "npaw-environment": "prod",
    }


def test_adapter_headers_raise_when_missing_required_settings():
    with pytest.raises(MCPConfigurationError):
        NpawAdapter({"api_key": "x", "account_code": "y"})._headers()


def test_exception_redacts_runtime_credential():
    message = redact_exception(
        RuntimeError("request rejected runtime-value"), ("runtime-value",)
    )
    assert "runtime-value" not in message
    assert "***" in message


@pytest.mark.asyncio
async def test_healthy_user_uses_one_query_and_never_calls_other_tools():
    payload = {
        "status": "success",
        "data": "session_root,views,errors,bufferRatio\nsession-a,1,0,0\n",
    }

    class FakeSession:
        def __init__(self):
            self.calls = []

        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return result_with(payload)

    adapter = NpawAdapter(
        {
            "url": "https://mcp.example.invalid/sse",
            "api_key": "runtime-value",
            "account_code": "incidnet",
        }
    )
    session = FakeSession()
    result = await adapter.query_user(session, "user-101")
    assert len(session.calls) == 1
    assert session.calls[0][0] == "npaw_query_data"
    assert result["summary"]["sessions"] == 1


def test_incident_scope_keeps_only_roku_rows():
    rows = [
        {
            "session_root": "roku-session",
            "Extraparam8": "roku_4k_hdr",
            "Player": "",
            "errors": "0",
        },
        {
            "session_root": "desktop-session",
            "Extraparam8": "web",
            "Player": "clappr-web",
            "errors": "8",
        },
    ]
    scoped, target = scope_rows_for_incident(
        rows, "Falha de playback", "O incidente ocorreu no Roku 4K."
    )
    assert target == "Roku"
    assert [row["session_root"] for row in scoped] == ["roku-session"]


def test_incident_scope_uses_player_name_for_html_tv():
    rows = [
        {"player_name": "clappr-web", "session_root": "desktop"},
        {"player_name": "clappr-web-tvs", "session_root": "html-tv"},
        {"player_name": "clappr-native-tvs", "session_root": "native-tv"},
    ]
    scoped, target = scope_rows_for_incident(
        rows, "Erro em TVs HTML", "O usuário reportou falha na televisão."
    )
    assert target == "TV HTML"
    assert [row["session_root"] for row in scoped] == ["html-tv"]


def test_incident_scope_uses_native_tv_and_vendor_from_description():
    rows = [
        {
            "Player": "clappr-native-tvs",
            "Device": "LG webOS",
            "session_root": "lg-native",
        },
        {
            "Player": "clappr-web-tvs",
            "Device": "Samsung - Tizen",
            "session_root": "samsung-html",
        },
    ]
    scoped, target = scope_rows_for_incident(
        rows,
        "Incidente em TVs",
        "Usuários reclamando sobre consumo com TVs LG nativa.",
    )
    assert target == "TV nativa"
    assert [row["session_root"] for row in scoped] == ["lg-native"]
