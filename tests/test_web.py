from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app import main
from app.investigator import Investigator
from app.models import UserDiagnosis
from app.scheduler import InvestigationScheduler
from app.storage import FileStorage


def build_client(tmp_path, monkeypatch):
    storage = FileStorage(tmp_path / "workspace")
    investigator = Investigator(storage)
    scheduler = InvestigationScheduler(storage, investigator)
    monkeypatch.setattr(main, "storage", storage)
    monkeypatch.setattr(main, "investigator", investigator)
    monkeypatch.setattr(main, "scheduler", scheduler)
    return TestClient(main.app, follow_redirects=False), storage


def test_complete_ui_flow(tmp_path, monkeypatch):
    client, storage = build_client(tmp_path, monkeypatch)
    with client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Criar novo workspace" in response.text

        response = client.post(
            "/api/workspaces/create",
            data={
                "name": "Acme",
            },
        )
        assert response.status_code == 303
        workspace_id = response.headers["location"].split("/")[-1]

        response = client.post(
            f"/w/{workspace_id}/incidents/create",
            data={
                "title": "Playback",
                "description": "Falha observada no player Roku.",
                "affected_users": "usr_101\nusr_102,usr_101",
            },
        )
        assert response.status_code == 303
        incident_id = response.headers["location"].split("/")[-1]

        page = client.get(response.headers["location"])
        assert page.status_code == 200
        assert "Playback" in page.text
        assert "Falha observada no player Roku." in page.text
        assert "2 analisados" not in page.text

        updated = client.post(
            f"/w/{workspace_id}/i/{incident_id}/details",
            data={
                "title": "Playback no Roku",
                "description": "Somente dispositivos Roku são afetados.",
            },
        )
        assert updated.status_code == 303
        assert storage.get_incident(
            workspace_id, incident_id
        ).description == "Somente dispositivos Roku são afetados."

        storage.save_diagnosis(
            workspace_id,
            incident_id,
            UserDiagnosis(
                user_id="usr_101",
                classification="BAD",
                justification="Erros confirmados no device alvo.",
                investigated_at=datetime.now(timezone.utc),
                source="TEST",
                summary={"sessions": 2, "plays": "3"},
            ),
        )
        storage.save_diagnosis(
            workspace_id,
            incident_id,
            UserDiagnosis(
                user_id="usr_102",
                classification="GOOD",
                justification="Sem falhas no device alvo.",
                investigated_at=datetime.now(timezone.utc),
                source="TEST",
                summary={"sessions": 1, "plays": "1"},
            ),
        )

        bad_page = client.get(
            f"/w/{workspace_id}/i/{incident_id}/diagnoses/BAD"
        )
        assert bad_page.status_code == 200
        assert "Erros confirmados no device alvo." in bad_page.text
        assert "Sem falhas no device alvo." not in bad_page.text

        good_page = client.get(
            f"/w/{workspace_id}/i/{incident_id}/diagnoses/GOOD"
        )
        assert good_page.status_code == 200
        assert "Sem falhas no device alvo." in good_page.text
        assert client.get(
            f"/w/{workspace_id}/i/{incident_id}/diagnoses/unknown"
        ).status_code == 404

        comment = client.post(
            f"/w/{workspace_id}/i/{incident_id}/comment",
            data={"author": "N2", "content": "**Regra nova**"},
        )
        assert comment.status_code == 303
        assert storage.get_feed(workspace_id, incident_id).entries[-1].author == "N2"

        removed_user = client.post(
            f"/w/{workspace_id}/i/{incident_id}/users/usr_101/delete"
        )
        assert removed_user.status_code == 303
        assert storage.get_incident(
            workspace_id, incident_id
        ).affected_users == ["usr_102"]

        removed_incident = client.post(
            f"/w/{workspace_id}/i/{incident_id}/delete"
        )
        assert removed_incident.status_code == 303
        assert storage.list_incidents(workspace_id) == []


def test_resolve_and_reopen_user(tmp_path, monkeypatch):
    client, storage = build_client(tmp_path, monkeypatch)
    with client:
        client.post("/api/workspaces/create", data={"name": "Acme"})
        workspace_id = storage.list_workspaces()[0].workspace_id
        resp = client.post(
            f"/w/{workspace_id}/incidents/create",
            data={"title": "Inc", "affected_users": "usr_10"},
        )
        incident_id = resp.headers["location"].split("/")[-1]

        storage.save_diagnosis(
            workspace_id,
            incident_id,
            UserDiagnosis(
                user_id="usr_10",
                classification="BAD",
                justification="Erros.",
                investigated_at=datetime.now(timezone.utc),
                source="TEST",
            ),
        )

        resolve_resp = client.post(
            f"/w/{workspace_id}/i/{incident_id}/users/usr_10/resolve",
            data={"resolved_by": "analista"},
        )
        assert resolve_resp.status_code == 303
        diagnosis = storage.list_diagnoses(workspace_id, incident_id)[0]
        assert diagnosis.resolved is True
        assert diagnosis.resolved_by == "analista"
        assert diagnosis.status == "RESOLVED"
        assert storage.get_incident(workspace_id, incident_id).status == "RESOLVED"

        reopen_resp = client.post(
            f"/w/{workspace_id}/i/{incident_id}/users/usr_10/reopen"
        )
        assert reopen_resp.status_code == 303
        diagnosis = storage.list_diagnoses(workspace_id, incident_id)[0]
        assert diagnosis.resolved is False
        assert diagnosis.status == "BAD"
        assert storage.get_incident(workspace_id, incident_id).status == "OPEN"


def test_mentioning_agent_triggers_reply_and_memory(tmp_path, monkeypatch):
    import app.investigator as investigator_module

    async def fake_answer_mention(**kwargs):
        return "Resposta do agente."

    async def fake_extract_memory_note(**kwargs):
        return "Para esse incidente, olhe o bitrate."

    monkeypatch.setattr(investigator_module, "answer_mention", fake_answer_mention)
    monkeypatch.setattr(
        investigator_module, "extract_memory_note", fake_extract_memory_note
    )

    client, storage = build_client(tmp_path, monkeypatch)
    with client:
        client.post("/api/workspaces/create", data={"name": "Acme"})
        workspace_id = storage.list_workspaces()[0].workspace_id
        resp = client.post(
            f"/w/{workspace_id}/incidents/create",
            data={"title": "Inc", "affected_users": "usr_10"},
        )
        incident_id = resp.headers["location"].split("/")[-1]

        comment = client.post(
            f"/w/{workspace_id}/i/{incident_id}/comment",
            data={"author": "N2", "content": "@agente o que aconteceu?"},
        )
        assert comment.status_code == 303

        feed = storage.get_feed(workspace_id, incident_id)
        agent_comments = [
            entry
            for entry in feed.entries
            if entry.author_type == "AGENT" and entry.kind == "COMMENT"
        ]
        assert agent_comments and agent_comments[-1].content == "Resposta do agente."

        memory = storage.get_incident_memory(workspace_id, incident_id)
        assert len(memory.entries) == 1
        assert "bitrate" in memory.entries[0].content


def test_plain_comment_does_not_trigger_agent(tmp_path, monkeypatch):
    import app.investigator as investigator_module

    called = False

    async def fake_answer_mention(**kwargs):
        nonlocal called
        called = True
        return "x"

    monkeypatch.setattr(investigator_module, "answer_mention", fake_answer_mention)

    client, storage = build_client(tmp_path, monkeypatch)
    with client:
        client.post("/api/workspaces/create", data={"name": "Acme"})
        workspace_id = storage.list_workspaces()[0].workspace_id
        resp = client.post(
            f"/w/{workspace_id}/incidents/create",
            data={"title": "Inc", "affected_users": "usr_10"},
        )
        incident_id = resp.headers["location"].split("/")[-1]

        client.post(
            f"/w/{workspace_id}/i/{incident_id}/comment",
            data={"author": "N2", "content": "Só um comentário normal."},
        )

    assert called is False


def test_diagnoses_resolved_filter(tmp_path, monkeypatch):
    client, storage = build_client(tmp_path, monkeypatch)
    with client:
        client.post("/api/workspaces/create", data={"name": "Acme"})
        workspace_id = storage.list_workspaces()[0].workspace_id
        resp = client.post(
            f"/w/{workspace_id}/incidents/create",
            data={"title": "Inc", "affected_users": "usr_20"},
        )
        incident_id = resp.headers["location"].split("/")[-1]

        storage.save_diagnosis(
            workspace_id,
            incident_id,
            UserDiagnosis(
                user_id="usr_20",
                classification="BAD",
                justification="Falha confirmada.",
                investigated_at=datetime.now(timezone.utc),
                source="TEST",
            ),
        )
        storage.mark_user_resolved(
            workspace_id, incident_id, "usr_20", resolved_by="analista"
        )

        page = client.get(f"/w/{workspace_id}/i/{incident_id}/diagnoses/RESOLVED")
        assert page.status_code == 200
        assert "Falha confirmada." in page.text
        assert "analista" in page.text

        # Must not appear under BAD anymore
        bad_page = client.get(f"/w/{workspace_id}/i/{incident_id}/diagnoses/BAD")
        assert "Falha confirmada." not in bad_page.text


def test_export_users_csv_and_xlsx(tmp_path, monkeypatch):
    client, storage = build_client(tmp_path, monkeypatch)
    with client:
        client.post("/api/workspaces/create", data={"name": "Acme"})
        workspace_id = storage.list_workspaces()[0].workspace_id
        resp = client.post(
            f"/w/{workspace_id}/incidents/create",
            data={"title": "Inc", "affected_users": "usr_30"},
        )
        incident_id = resp.headers["location"].split("/")[-1]

        storage.save_diagnosis(
            workspace_id,
            incident_id,
            UserDiagnosis(
                user_id="usr_30",
                classification="BAD",
                justification="Falha confirmada no device alvo.",
                investigated_at=datetime.now(timezone.utc),
                source="TEST",
                summary={"sessions": 1},
            ),
        )

        csv_resp = client.get(f"/w/{workspace_id}/i/{incident_id}/export?fmt=csv")
        assert csv_resp.status_code == 200
        assert csv_resp.headers["content-type"].startswith("text/csv")
        assert "attachment" in csv_resp.headers["content-disposition"]
        assert "user_id,classification,status" in csv_resp.text
        assert "usr_30,BAD,BAD" in csv_resp.text

        xlsx_resp = client.get(f"/w/{workspace_id}/i/{incident_id}/export?fmt=xlsx")
        assert xlsx_resp.status_code == 200
        assert xlsx_resp.headers["content-type"] == (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert len(xlsx_resp.content) > 0

        invalid_resp = client.get(f"/w/{workspace_id}/i/{incident_id}/export?fmt=pdf")
        assert invalid_resp.status_code == 400

        missing_resp = client.get(f"/w/{workspace_id}/i/inc_missing/export?fmt=csv")
        assert missing_resp.status_code == 404


def test_health_endpoint(tmp_path, monkeypatch):
    client, _ = build_client(tmp_path, monkeypatch)
    with client:
        response = client.get("/health")
        assert response.json()["status"] == "ok"


def test_agent_guide_endpoint(tmp_path, monkeypatch):
    client, _ = build_client(tmp_path, monkeypatch)
    with client:
        response = client.get("/api/agent/guide")
        assert response.status_code == 200
        body = response.json()
        assert "/api/workspaces" in body["instructions"]
        assert body["openapi"].endswith("/openapi.json")

        md = client.get("/api/agent/guide?format=md")
        assert md.status_code == 200
        assert md.headers["content-type"].startswith("text/markdown")
        assert "# Guia de integração" in md.text


def test_agent_json_api_full_flow(tmp_path, monkeypatch):
    client, storage = build_client(tmp_path, monkeypatch)
    with client:
        created = client.post("/api/workspaces", json={"name": "Acme JSON"})
        assert created.status_code == 201
        workspace_id = created.json()["workspace_id"]

        settings = client.post(
            f"/api/workspaces/{workspace_id}/settings",
            json={"cron_schedule": "0 8 * * *"},
        )
        assert settings.status_code == 200
        assert settings.json()["cron_schedule"] == "0 8 * * *"

        bad_cron = client.post(
            f"/api/workspaces/{workspace_id}/settings",
            json={"cron_schedule": "not-a-cron"},
        )
        assert bad_cron.status_code == 400

        incident = client.post(
            f"/api/workspaces/{workspace_id}/incidents",
            json={
                "title": "Playback Roku",
                "description": "Erros no player.",
                "affected_users": ["usr_1", "usr_2"],
            },
        )
        assert incident.status_code == 201
        incident_id = incident.json()["incident_id"]
        assert incident.json()["affected_users"] == ["usr_1", "usr_2"]

        storage.save_diagnosis(
            workspace_id,
            incident_id,
            UserDiagnosis(
                user_id="usr_1",
                classification="BAD",
                justification="Erros confirmados.",
                investigated_at=datetime.now(timezone.utc),
                source="TEST",
            ),
        )

        detail = client.get(f"/api/workspaces/{workspace_id}/incidents/{incident_id}")
        assert detail.status_code == 200
        payload = detail.json()
        assert payload["counts"]["BAD"] == 1
        assert payload["diagnoses"][0]["status"] == "BAD"

        resolved = client.post(
            f"/api/workspaces/{workspace_id}/incidents/{incident_id}/users/usr_1/resolve",
            json={"resolved_by": "meu-agente"},
        )
        assert resolved.status_code == 200
        assert resolved.json()["status"] == "RESOLVED"

        resolved_list = client.get(
            f"/api/workspaces/{workspace_id}/incidents/{incident_id}/diagnoses",
            params={"status_name": "RESOLVED"},
        )
        assert resolved_list.status_code == 200
        assert len(resolved_list.json()) == 1

        deleted = client.delete(
            f"/api/workspaces/{workspace_id}/incidents/{incident_id}"
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True


def test_external_agent_can_publish_and_read_analysis(tmp_path, monkeypatch):
    client, storage = build_client(tmp_path, monkeypatch)
    with client:
        workspace = client.post("/api/workspaces", json={"name": "Acme"}).json()
        incident = client.post(
            f"/api/workspaces/{workspace['workspace_id']}/incidents",
            json={"title": "Playback", "affected_users": ["usr_1"]},
        ).json()
        base = f"/api/workspaces/{workspace['workspace_id']}/incidents/{incident['incident_id']}"
        created = client.post(
            f"{base}/analyses",
            json={
                "type": "code_investigation",
                "agent_name": "Code Agent",
                "content": "A versão 2.4 introduziu uma regressão no player.",
                "diagnoses": [{
                    "user_id": "usr_1",
                    "classification": "BAD",
                    "justification": "Erro de inicialização após o deploy.",
                }],
            },
        )
        assert created.status_code == 201
        analysis_id = created.json()["analysis_id"]
        assert created.json()["summary"] == {
            "total_users": 1, "good": 0, "bad": 1, "inconclusive": 0,
        }
        assert client.get(f"{base}/analyses/latest").json()["analysis_id"] == analysis_id
        detail = client.get(f"{base}/analyses/{analysis_id}")
        assert detail.status_code == 200
        assert detail.json()["diagnoses"][0]["classification"] == "BAD"
        page = client.get(
            f"/w/{workspace['workspace_id']}/i/{incident['incident_id']}"
        )
        assert page.status_code == 200
        assert analysis_id in page.text
        analysis_page = client.get(
            f"/w/{workspace['workspace_id']}/i/{incident['incident_id']}/analyses/{analysis_id}"
        )
        assert analysis_page.status_code == 200
        assert "regressão no player" in analysis_page.text
        assert "Análise" in storage.get_feed(
            workspace["workspace_id"], incident["incident_id"]
        ).entries[-1].content
