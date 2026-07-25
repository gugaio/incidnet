from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

import app.investigator as investigator_module
from app.investigator import Investigator
from app.models import UserDiagnosis
from app.storage import FileStorage


@pytest.mark.asyncio
async def test_missing_mcp_configuration_is_reported_in_feed(tmp_path, monkeypatch):
    for name in ("NPAW_MCP_URL", "NPAW_ACCOUNT_CODE", "NPAW_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Incidnet Test")
    incident = storage.create_incident(
        workspace.workspace_id, "Incident Test", ["user-101"]
    )

    await Investigator(storage).investigate(
        workspace.workspace_id, incident.incident_id
    )

    feed = storage.get_feed(workspace.workspace_id, incident.incident_id)
    assert feed.entries[-1].kind == "ERROR"
    assert "não configurada" in feed.entries[-1].content


class _FakeSession:
    async def list_tools(self):
        return SimpleNamespace(tools=[])


class _FakeAdapter:
    def secret(self) -> str:
        return ""

    def validate_config(self) -> bool:
        return False

    def required_tools(self) -> set[str]:
        return set()

    def classify(self, rows):
        return ("GOOD", "ok")

    @asynccontextmanager
    async def session(self):
        yield _FakeSession()


@pytest.mark.asyncio
async def test_investigate_runs_users_concurrently_and_tracks_progress(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        investigator_module, "get_adapter", lambda _config: _FakeAdapter()
    )

    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Acme")
    user_ids = [f"user-{i}" for i in range(20)]
    incident = storage.create_incident(workspace.workspace_id, "Inc", user_ids)
    investigator = Investigator(storage)

    async def fake_investigate_user(
        self, adapter, session, user_id, prompt, incident_title, incident_description,
        memory="",
    ):
        index = int(user_id.split("-")[1])
        classification = "BAD" if index % 5 == 0 else "GOOD"
        return UserDiagnosis(
            user_id=user_id,
            classification=classification,
            justification="x",
            investigated_at=storage.now(),
            source="TEST",
        )

    monkeypatch.setattr(Investigator, "_investigate_user", fake_investigate_user)

    assert investigator.progress(workspace.workspace_id, incident.incident_id) is None

    await investigator.investigate(workspace.workspace_id, incident.incident_id)

    diagnoses = storage.list_diagnoses(workspace.workspace_id, incident.incident_id)
    assert len(diagnoses) == len(user_ids)
    assert {item.user_id for item in diagnoses} == set(user_ids)

    progress = investigator.progress(workspace.workspace_id, incident.incident_id)
    assert progress["phase"] == "done"
    assert progress["done"] == len(user_ids)
    assert progress["total"] == len(user_ids)
    assert progress["in_flight"] == []
    assert progress["counts"] == {"BAD": 4, "GOOD": 16}

    feed = storage.get_feed(workspace.workspace_id, incident.incident_id)
    assert feed.entries[-1].kind == "DIAGNOSIS"
    assert "Análise concluída" in feed.entries[-1].content
    analyses = storage.list_analyses(workspace.workspace_id, incident.incident_id)
    assert len(analyses) == 1
    assert analyses[0].summary.good == 16
    assert analyses[0].summary.bad == 4
    assert len(storage.list_analysis_diagnoses(
        workspace.workspace_id, incident.incident_id, analyses[0].analysis_id
    )) == len(user_ids)


@pytest.mark.asyncio
async def test_respond_to_mention_replies_and_persists_learning(
    tmp_path, monkeypatch
):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Acme")
    incident = storage.create_incident(
        workspace.workspace_id, "Buffering na Smart TV", ["user-1"]
    )
    # Seed a completed diagnosis round in the feed.
    storage.append_feed(
        workspace.workspace_id,
        incident.incident_id,
        author="Agente Incidnet",
        author_type="AGENT",
        kind="DIAGNOSIS",
        content="### Rodada de diagnóstico concluída\n\nResultado GOOD: 100%",
    )

    monkeypatch.setattr(
        investigator_module, "get_adapter", lambda _cfg: _FakeAdapter()
    )

    async def fake_answer_mention(**kwargs):
        return "Resposta do agente."

    async def fake_extract_memory_note(**kwargs):
        return "Para buffering em Smart TV, verifique a métrica de bitrate."

    monkeypatch.setattr(
        investigator_module, "answer_mention", fake_answer_mention
    )
    monkeypatch.setattr(
        investigator_module, "extract_memory_note", fake_extract_memory_note
    )

    investigator = Investigator(storage)
    await investigator.respond_to_mention(
        workspace.workspace_id,
        incident.incident_id,
        "@agente o que achou?",
        author="Fulano",
    )

    feed = storage.get_feed(workspace.workspace_id, incident.incident_id)
    agent_replies = [
        entry for entry in feed.entries
        if entry.author_type == "AGENT" and entry.kind == "COMMENT"
    ]
    assert agent_replies and agent_replies[-1].content == "Resposta do agente."

    memory = storage.get_incident_memory(
        workspace.workspace_id, incident.incident_id
    )
    assert len(memory.entries) == 1
    assert "bitrate" in memory.entries[0].content


@pytest.mark.asyncio
async def test_respond_to_mention_without_learning_does_not_persist(
    tmp_path, monkeypatch
):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Acme")
    incident = storage.create_incident(
        workspace.workspace_id, "Incidente", ["user-1"]
    )

    monkeypatch.setattr(
        investigator_module, "get_adapter", lambda _cfg: _FakeAdapter()
    )

    async def fake_answer_mention(**kwargs):
        return "Sem novidades."

    async def fake_extract_memory_note(**kwargs):
        return None

    monkeypatch.setattr(
        investigator_module, "answer_mention", fake_answer_mention
    )
    monkeypatch.setattr(
        investigator_module, "extract_memory_note", fake_extract_memory_note
    )

    investigator = Investigator(storage)
    await investigator.respond_to_mention(
        workspace.workspace_id, incident.incident_id, "@agente obrigado", "Fulano"
    )

    memory = storage.get_incident_memory(
        workspace.workspace_id, incident.incident_id
    )
    assert memory.entries == []


@pytest.mark.asyncio
async def test_consolidate_memories_promotes_notes_to_workspace(
    tmp_path, monkeypatch
):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Acme")
    incident = storage.create_incident(
        workspace.workspace_id, "Incidente", ["user-1"]
    )
    storage.append_incident_memory(
        workspace.workspace_id,
        incident.incident_id,
        "Dica específica do incidente sobre bitrate.",
    )

    async def fake_consolidate(**kwargs):
        assert kwargs["incident_notes"]
        return ["Regra geral: verifique bitrate em incidentes de buffering."]

    monkeypatch.setattr(
        investigator_module, "consolidate_workspace_memory", fake_consolidate
    )

    investigator = Investigator(storage)
    await investigator.consolidate_memories()

    workspace_memory = storage.get_workspace_memory(workspace.workspace_id)
    assert len(workspace_memory.entries) == 1
    assert "bitrate" in workspace_memory.entries[0].content
