from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from app.storage import FileStorage, NotFoundError, StorageError
from app.models import UserDiagnosis


def test_workspace_and_incident_structure(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    storage.initialize()
    workspace = storage.create_workspace("Workspace Teste")
    incident = storage.create_incident(
        workspace.workspace_id,
        "Falha de playback",
        ["user-101", "user-102", "user-101"],
        description="Afeta o device descrito pelo atendimento.",
    )

    workspace_dir = tmp_path / "workspace" / "workspaces" / workspace.workspace_id
    incident_dir = workspace_dir / "incidents" / incident.incident_id
    assert (workspace_dir / "workspace.json").exists()
    assert (workspace_dir / "PROMPT_BASE.md").exists()
    assert (workspace_dir / "incidents" / "index.json").exists()
    assert (incident_dir / "incident.json").exists()
    assert (incident_dir / "feed.json").exists()
    assert (incident_dir / "users").is_dir()
    assert incident.affected_users == ["user-101", "user-102"]
    assert incident.description == "Afeta o device descrito pelo atendimento."

    feed = json.loads((incident_dir / "feed.json").read_text())
    assert feed["entries"][0]["author_type"] == "AGENT"


def test_path_traversal_is_rejected(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    storage.initialize()
    with pytest.raises(StorageError):
        storage.get_workspace("../../etc")


def test_workspace_file_never_contains_integration_secrets(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    storage.update_workspace(
        workspace.workspace_id,
        prompt="# Updated",
        cron_schedule="0 0 * * *",
    )
    workspace_file = (
        tmp_path
        / "workspace"
        / "workspaces"
        / workspace.workspace_id
        / "workspace.json"
    ).read_text()
    assert "api_key" not in workspace_file.lower()
    assert "auth_token" not in workspace_file.lower()
    assert "account_code" not in workspace_file.lower()
    assert "server_url" not in workspace_file.lower()


def test_remove_user_updates_incident_index_and_diagnosis(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(
        workspace.workspace_id, "Incidente", ["user-101", "user-102"]
    )
    diagnosis_path = (
        tmp_path
        / "workspace"
        / "workspaces"
        / workspace.workspace_id
        / "incidents"
        / incident.incident_id
        / "users"
        / "user-101.json"
    )
    diagnosis_path.write_text("{}")

    updated = storage.remove_user(
        workspace.workspace_id, incident.incident_id, "user-101"
    )

    assert updated.affected_users == ["user-102"]
    assert not diagnosis_path.exists()
    assert storage.list_incidents(workspace.workspace_id)[0].affected_users == [
        "user-102"
    ]


def test_removing_last_unresolved_user_resolves_incident(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(
        workspace.workspace_id, "Incidente", ["user-101", "user-102"]
    )
    storage.save_diagnosis(
        workspace.workspace_id, incident.incident_id, _make_diagnosis("user-102")
    )
    storage.mark_user_resolved(
        workspace.workspace_id, incident.incident_id, "user-102", resolved_by="N2"
    )
    assert storage.get_incident(workspace.workspace_id, incident.incident_id).status == "OPEN"

    storage.remove_user(workspace.workspace_id, incident.incident_id, "user-101")

    assert (
        storage.get_incident(workspace.workspace_id, incident.incident_id).status
        == "RESOLVED"
    )


def test_delete_incident_removes_only_selected_incident(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    first = storage.create_incident(
        workspace.workspace_id, "Primeiro", ["user-101"]
    )
    second = storage.create_incident(
        workspace.workspace_id, "Segundo", ["user-102"]
    )

    storage.delete_incident(workspace.workspace_id, first.incident_id)

    assert [item.incident_id for item in storage.list_incidents(workspace.workspace_id)] == [
        second.incident_id
    ]
    with pytest.raises(NotFoundError):
        storage.get_incident(workspace.workspace_id, first.incident_id)


def test_initialize_migrates_legacy_diagnosis_status(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(
        workspace.workspace_id, "Incidente", ["user-101"]
    )
    users_dir = (
        tmp_path
        / "workspace"
        / "workspaces"
        / workspace.workspace_id
        / "incidents"
        / incident.incident_id
        / "users"
    )
    legacy = UserDiagnosis(
        user_id="user-101",
        classification="GOOD",
        justification="Sem falhas",
        investigated_at=datetime.now(timezone.utc),
        source="TEST",
    ).model_dump(mode="json")
    legacy["classification"] = "HEALTHY"
    (users_dir / "user-101.json").write_text(json.dumps(legacy))

    storage.initialize()

    migrated = json.loads((users_dir / "user-101.json").read_text())
    assert migrated["classification"] == "GOOD"


def _make_diagnosis(user_id: str = "user-101") -> UserDiagnosis:
    return UserDiagnosis(
        user_id=user_id,
        classification="BAD",
        justification="Erros detectados",
        investigated_at=datetime.now(timezone.utc),
        source="TEST",
    )


def test_save_diagnosis_preserves_resolved_state(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(workspace.workspace_id, "Inc", ["user-101"])

    storage.save_diagnosis(workspace.workspace_id, incident.incident_id, _make_diagnosis())
    storage.mark_user_resolved(
        workspace.workspace_id, incident.incident_id, "user-101", resolved_by="N2"
    )

    # Re-investigation saves a new diagnosis — resolved state must be preserved
    storage.save_diagnosis(workspace.workspace_id, incident.incident_id, _make_diagnosis())

    diagnosis = storage.list_diagnoses(workspace.workspace_id, incident.incident_id)[0]
    assert diagnosis.resolved is True
    assert diagnosis.resolved_by == "N2"
    assert diagnosis.resolved_at is not None


def test_mark_user_resolved(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(workspace.workspace_id, "Inc", ["user-101"])
    storage.save_diagnosis(workspace.workspace_id, incident.incident_id, _make_diagnosis())

    result = storage.mark_user_resolved(
        workspace.workspace_id, incident.incident_id, "user-101", resolved_by="time-n3"
    )

    assert result.resolved is True
    assert result.resolved_by == "time-n3"
    assert result.resolved_at is not None
    assert result.status == "RESOLVED"
    # classification (raw) must remain unchanged
    assert result.classification == "BAD"


def test_incident_becomes_resolved_only_when_every_user_is_resolved(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(
        workspace.workspace_id, "Inc", ["user-101", "user-102"]
    )
    storage.save_diagnosis(
        workspace.workspace_id, incident.incident_id, _make_diagnosis("user-101")
    )
    storage.save_diagnosis(
        workspace.workspace_id, incident.incident_id, _make_diagnosis("user-102")
    )

    storage.mark_user_resolved(
        workspace.workspace_id, incident.incident_id, "user-101", resolved_by="N2"
    )
    assert storage.get_incident(workspace.workspace_id, incident.incident_id).status == "OPEN"

    storage.mark_user_resolved(
        workspace.workspace_id, incident.incident_id, "user-102", resolved_by="N2"
    )
    assert (
        storage.get_incident(workspace.workspace_id, incident.incident_id).status
        == "RESOLVED"
    )

    storage.reopen_user(workspace.workspace_id, incident.incident_id, "user-101")
    assert storage.get_incident(workspace.workspace_id, incident.incident_id).status == "OPEN"


def test_mark_user_resolved_raises_when_no_diagnosis(tmp_path):
    from app.storage import NotFoundError

    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(workspace.workspace_id, "Inc", ["user-101"])

    with pytest.raises(NotFoundError):
        storage.mark_user_resolved(
            workspace.workspace_id, incident.incident_id, "user-101", resolved_by="N2"
        )


def test_reopen_user(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(workspace.workspace_id, "Inc", ["user-101"])
    storage.save_diagnosis(workspace.workspace_id, incident.incident_id, _make_diagnosis())
    storage.mark_user_resolved(
        workspace.workspace_id, incident.incident_id, "user-101", resolved_by="N2"
    )

    result = storage.reopen_user(workspace.workspace_id, incident.incident_id, "user-101")

    assert result.resolved is False
    assert result.resolved_by is None
    assert result.resolved_at is None
    assert result.status == "BAD"


def test_incident_and_workspace_memory_roundtrip_and_context(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(workspace.workspace_id, "Inc", ["user-101"])

    # Empty memory returns empty context.
    assert storage.build_memory_context(workspace.workspace_id, incident.incident_id) == ""
    assert storage.get_incident_memory(workspace.workspace_id, incident.incident_id).entries == []

    entry = storage.append_incident_memory(
        workspace.workspace_id, incident.incident_id, "Olhe a métrica de bitrate."
    )
    assert entry is not None
    storage.append_workspace_memory(
        workspace.workspace_id, "Regra geral de buffering."
    )

    context = storage.build_memory_context(
        workspace.workspace_id, incident.incident_id
    )
    assert "bitrate" in context
    assert "buffering" in context
    assert "workspace" in context.lower()


def test_append_memory_deduplicates(tmp_path):
    storage = FileStorage(tmp_path / "workspace")
    workspace = storage.create_workspace("Workspace")
    incident = storage.create_incident(workspace.workspace_id, "Inc", ["user-101"])

    first = storage.append_incident_memory(
        workspace.workspace_id, incident.incident_id, "Verifique o bitrate."
    )
    duplicate = storage.append_incident_memory(
        workspace.workspace_id, incident.incident_id, "  verifique   o  BITRATE.  "
    )

    assert first is not None
    assert duplicate is None
    memory = storage.get_incident_memory(
        workspace.workspace_id, incident.incident_id
    )
    assert len(memory.entries) == 1
