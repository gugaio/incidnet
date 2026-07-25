from __future__ import annotations

import pytest

from app.adapters.npaw import deterministic_classification
from app.llm import _deterministic_result, analyze_telemetry


def test_deterministic_result_mentions_target_device():
    result = _deterministic_result(
        {
            "rows": [
                {
                    "player_type": "roku",
                    "errors": "0",
                    "bufferRatio": "0",
                }
            ],
            "scope": {"target_device": "Roku"},
        },
        deterministic_classification,
    )
    assert result.classification == "GOOD"
    assert result.justification.startswith("Escopo Roku:")


def test_deterministic_result_reports_no_rows_for_target_device():
    result = _deterministic_result(
        {
            "rows": [],
            "scope": {"target_device": "TV HTML", "total_rows": 2},
        },
        deterministic_classification,
    )
    assert result.classification == "INCONCLUSIVE"
    assert result.justification == (
        "Foram encontradas 2 rows de telemetria no período, mas nenhuma é "
        "compatível com o device alvo: TV HTML."
    )


@pytest.mark.asyncio
async def test_no_matching_rows_never_calls_llm_or_returns_unhealthy(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "configured-at-runtime")
    result, source = await analyze_telemetry(
        user_id="user-101",
        incident_title="Falha em TV",
        incident_description="TV nativa",
        prompt_base="# Prompt",
        telemetry={
            "rows": [],
            "scope": {"target_device": "TV nativa"},
        },
        classify=deterministic_classification,
    )
    assert result.classification == "INCONCLUSIVE"
    assert source == "DETERMINISTIC_NO_DATA"
