from __future__ import annotations

import csv
import io
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator, Literal

import bleach
import markdown
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from openpyxl import Workbook
from pydantic import BaseModel, Field

from .adapters import get_adapter, load_env_mcp_config
from .config import TEMPLATES_ROOT
from .investigator import Investigator
from .log import get_logger, setup_logging
from .models import UserDiagnosis
from .scheduler import InvestigationScheduler
from .storage import FileStorage, NotFoundError, StorageError

logger = get_logger(__name__)


storage = FileStorage()
investigator = Investigator(storage)
scheduler = InvestigationScheduler(storage, investigator)
templates = Jinja2Templates(directory=str(TEMPLATES_ROOT))


def render_markdown(value: str) -> Markup:
    html = markdown.markdown(
        value,
        extensions=["fenced_code", "tables", "sane_lists"],
        output_format="html",
    )
    clean = bleach.clean(
        html,
        tags={
            "p", "br", "strong", "em", "ul", "ol", "li", "code", "pre",
            "blockquote", "h1", "h2", "h3", "h4", "table", "thead", "tbody",
            "tr", "th", "td", "a",
        },
        attributes={"a": ["href", "title"]},
        protocols={"http", "https", "mailto"},
    )
    return Markup(clean)


templates.env.filters["markdown"] = render_markdown


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    logger.info("Incidnet iniciando", extra={"version": app.version})
    storage.initialize()
    scheduler.start()
    logger.info("Incidnet pronto", extra={"workspaces": len(storage.list_workspaces())})
    yield
    logger.info("Incidnet encerrando")
    scheduler.shutdown()


app = FastAPI(
    title="Incidnet",
    version="1.0.0",
    description="Orquestração e investigação agent-first de incidentes N2/N3.",
    lifespan=lifespan,
)


@app.exception_handler(NotFoundError)
async def not_found_handler(request: Request, exc: NotFoundError) -> HTMLResponse:
    logger.warning("NotFoundError", extra={"path": request.url.path, "error": str(exc)})
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={"message": str(exc), "status_code": 404},
        status_code=404,
    )


@app.exception_handler(StorageError)
async def storage_error_handler(request: Request, exc: StorageError) -> HTMLResponse:
    logger.warning("StorageError", extra={"path": request.url.path, "error": str(exc)})
    return templates.TemplateResponse(
        request=request,
        name="error.html",
        context={"message": str(exc), "status_code": 400},
        status_code=400,
    )


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="landing.html",
        context={"workspaces": storage.list_workspaces()},
    )


@app.post("/api/workspaces/create")
async def create_workspace(
    name: Annotated[str, Form()],
) -> RedirectResponse:
    workspace = storage.create_workspace(name)
    scheduler.refresh()
    logger.info("Workspace criado", extra={"workspace_id": workspace.workspace_id, "workspace_name": name})
    return RedirectResponse(
        f"/w/{workspace.workspace_id}", status_code=status.HTTP_303_SEE_OTHER
    )


@app.get("/w/{workspace_id}", response_class=HTMLResponse)
async def workspace_dashboard(request: Request, workspace_id: str) -> HTMLResponse:
    workspace = storage.get_workspace(workspace_id)
    return templates.TemplateResponse(
        request=request,
        name="workspace.html",
        context={
            "workspace": workspace,
            "incidents": storage.list_incidents(workspace_id),
            "prompt_base": storage.get_prompt(workspace_id),
            "mcp_configured": get_adapter(load_env_mcp_config()).validate_config(),
            "llm_configured": all(
                os.getenv(name, "").strip()
                for name in ("OPENAI_API_KEY", "OPENAI_MODEL")
            ),
        },
    )


@app.post("/w/{workspace_id}/settings")
async def update_settings(
    workspace_id: str,
    prompt_base: Annotated[str, Form()],
    cron_schedule: Annotated[str, Form()] = "0 0 * * *",
) -> RedirectResponse:
    _validate_cron(cron_schedule)
    storage.update_workspace(
        workspace_id,
        prompt=prompt_base,
        cron_schedule=cron_schedule,
    )
    scheduler.refresh()
    return RedirectResponse(
        f"/w/{workspace_id}?saved=1", status_code=status.HTTP_303_SEE_OTHER
    )


def parse_user_ids(raw: str) -> list[str]:
    return [
        value.strip()
        for value in re.split(r"[,\n\r;]+", raw)
        if value.strip()
    ]


_AGENT_MENTION = re.compile(r"@(agente|agent|incidnet)\b", re.IGNORECASE)


def mentions_agent(content: str) -> bool:
    return bool(_AGENT_MENTION.search(content or ""))


@app.post("/w/{workspace_id}/incidents/create")
async def create_incident(
    workspace_id: str,
    title: Annotated[str, Form()],
    affected_users: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
) -> RedirectResponse:
    incident = storage.create_incident(
        workspace_id,
        title,
        parse_user_ids(affected_users),
        description=description,
    )
    logger.info(
        "Incidente criado",
        extra={
            "workspace_id": workspace_id,
            "incident_id": incident.incident_id,
            "title": title,
            "users": len(incident.affected_users),
        },
    )
    return RedirectResponse(
        f"/w/{workspace_id}/i/{incident.incident_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.get("/w/{workspace_id}/i/{incident_id}", response_class=HTMLResponse)
async def incident_feed(
    request: Request, workspace_id: str, incident_id: str
) -> HTMLResponse:
    workspace = storage.get_workspace(workspace_id)
    incident = storage.get_incident(workspace_id, incident_id)
    feed = storage.get_feed(workspace_id, incident_id)
    diagnoses = storage.list_diagnoses(workspace_id, incident_id)
    diagnoses_by_user = {item.user_id: item for item in diagnoses}
    analyses = storage.list_analyses(workspace_id, incident_id)
    latest = analyses[0] if analyses else None
    counts = ({
        "GOOD": latest.summary.good,
        "BAD": latest.summary.bad,
        "INCONCLUSIVE": latest.summary.inconclusive,
        "RESOLVED": sum(item.status == "RESOLVED" for item in diagnoses),
    } if latest else {
        key: sum(item.status == key for item in diagnoses)
        for key in ("GOOD", "BAD", "INCONCLUSIVE", "RESOLVED")
    })
    health = (
        round((counts["GOOD"] + counts["RESOLVED"]) / len(incident.affected_users) * 100)
        if incident.affected_users
        else 0
    )
    return templates.TemplateResponse(
        request=request,
        name="incident.html",
        context={
            "workspace": workspace,
            "incident": incident,
            "entries": list(reversed(feed.entries)),
            "diagnoses": diagnoses,
            "diagnoses_by_user": diagnoses_by_user,
            "analyses": analyses,
            "counts": counts,
            "health": health,
            "incident_memory": storage.get_incident_memory(
                workspace_id, incident_id
            ).entries,
            "workspace_memory": storage.get_workspace_memory(workspace_id).entries,
        },
    )


@app.get("/w/{workspace_id}/i/{incident_id}/analyses/{analysis_id}", response_class=HTMLResponse)
async def analysis_detail(
    request: Request, workspace_id: str, incident_id: str, analysis_id: str
) -> HTMLResponse:
    workspace = storage.get_workspace(workspace_id)
    incident = storage.get_incident(workspace_id, incident_id)
    analysis = storage.get_analysis(workspace_id, incident_id, analysis_id)
    return templates.TemplateResponse(
        request=request,
        name="analysis.html",
        context={
            "workspace": workspace,
            "incident": incident,
            "analysis": analysis,
            "diagnoses": storage.list_analysis_diagnoses(workspace_id, incident_id, analysis_id),
        },
    )


@app.get("/w/{workspace_id}/i/{incident_id}/export")
async def export_incident_users(workspace_id: str, incident_id: str, fmt: str = "xlsx") -> Response:
    fmt = fmt.lower()
    if fmt not in {"xlsx", "csv"}:
        raise HTTPException(status_code=400, detail="Formato inválido. Use 'xlsx' ou 'csv'.")
    storage.ensure_incident_exists(workspace_id, incident_id)
    diagnoses = storage.list_diagnoses(workspace_id, incident_id)
    rows = _export_rows(diagnoses)
    filename = f"incident_{incident_id}_users.{fmt}"

    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Usuários"
    sheet.append(EXPORT_COLUMNS)
    for row in rows:
        sheet.append([row[column] for column in EXPORT_COLUMNS])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return Response(
        content=buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get(
    "/w/{workspace_id}/i/{incident_id}/diagnoses/{status_name}",
    response_class=HTMLResponse,
)
async def diagnoses_by_status(
    request: Request,
    workspace_id: str,
    incident_id: str,
    status_name: str,
) -> HTMLResponse:
    selected_status = status_name.upper()
    allowed = {"GOOD", "BAD", "INCONCLUSIVE", "RESOLVED"}
    if selected_status not in allowed:
        raise HTTPException(status_code=404, detail="Status não encontrado")
    workspace = storage.get_workspace(workspace_id)
    incident = storage.get_incident(workspace_id, incident_id)
    all_diagnoses = storage.list_diagnoses(workspace_id, incident_id)
    diagnoses = [
        item
        for item in all_diagnoses
        if item.status == selected_status
    ]
    counts = {
        key: sum(item.status == key for item in all_diagnoses)
        for key in ("GOOD", "BAD", "INCONCLUSIVE", "RESOLVED")
    }
    status_meta = {
        "GOOD": {
            "label": "Sem falha",
            "description": (
                "Há dados suficientes do device alvo e nenhuma evidência de falha."
            ),
            "color": "emerald",
        },
        "BAD": {
            "label": "Com falha",
            "description": (
                "Há evidência de erro ou degradação em sessões do device alvo."
            ),
            "color": "rose",
        },
        "INCONCLUSIVE": {
            "label": "Sem evidência suficiente",
            "description": (
                "Não foi possível avaliar o device alvo com os dados disponíveis."
            ),
            "color": "amber",
        },
        "RESOLVED": {
            "label": "Resolvidos",
            "description": (
                "Usuários cujo problema foi tratado e marcados como resolvidos."
            ),
            "color": "violet",
        },
    }
    return templates.TemplateResponse(
        request=request,
        name="diagnoses.html",
        context={
            "workspace": workspace,
            "incident": incident,
            "diagnoses": diagnoses,
            "counts": counts,
            "selected_status": selected_status,
            "selected_meta": status_meta[selected_status],
            "status_meta": status_meta,
        },
    )


@app.post("/w/{workspace_id}/i/{incident_id}/users/{user_id}/resolve")
async def resolve_user(
    workspace_id: str,
    incident_id: str,
    user_id: str,
    resolved_by: Annotated[str, Form()] = "Equipe N2/N3",
) -> RedirectResponse:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError(
            "Aguarde a investigação atual terminar antes de marcar como resolvido"
        )
    storage.mark_user_resolved(
        workspace_id, incident_id, user_id, resolved_by=resolved_by
    )
    storage.append_feed(
        workspace_id,
        incident_id,
        author=resolved_by,
        author_type="HUMAN",
        kind="SYSTEM",
        content=f"Usuário `{user_id}` marcado como **resolvido** por {resolved_by}.",
    )
    return RedirectResponse(
        f"/w/{workspace_id}/i/{incident_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/w/{workspace_id}/i/{incident_id}/users/{user_id}/reopen")
async def reopen_user(
    workspace_id: str, incident_id: str, user_id: str
) -> RedirectResponse:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError(
            "Aguarde a investigação atual terminar antes de reabrir"
        )
    storage.reopen_user(workspace_id, incident_id, user_id)
    storage.append_feed(
        workspace_id,
        incident_id,
        author="Agente Incidnet",
        author_type="AGENT",
        kind="SYSTEM",
        content=f"Usuário `{user_id}` reaberto para monitoramento.",
    )
    return RedirectResponse(
        f"/w/{workspace_id}/i/{incident_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/w/{workspace_id}/i/{incident_id}/users/{user_id}/delete")
async def delete_incident_user(
    workspace_id: str, incident_id: str, user_id: str
) -> RedirectResponse:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError(
            "Aguarde a investigação atual terminar antes de remover usuários"
        )
    storage.remove_user(workspace_id, incident_id, user_id)
    storage.append_feed(
        workspace_id,
        incident_id,
        author="Agente Incidnet",
        author_type="AGENT",
        kind="SYSTEM",
        content=f"O usuário `{user_id}` foi removido do monitoramento.",
    )
    return RedirectResponse(
        f"/w/{workspace_id}/i/{incident_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/w/{workspace_id}/i/{incident_id}/details")
async def update_incident_details(
    workspace_id: str,
    incident_id: str,
    title: Annotated[str, Form()],
    description: Annotated[str, Form()] = "",
) -> RedirectResponse:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError(
            "Aguarde a investigação atual terminar antes de alterar o contexto"
        )
    storage.update_incident_details(
        workspace_id,
        incident_id,
        title=title,
        description=description,
    )
    return RedirectResponse(
        f"/w/{workspace_id}/i/{incident_id}?saved=1",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/w/{workspace_id}/i/{incident_id}/delete")
async def delete_incident(
    workspace_id: str, incident_id: str
) -> RedirectResponse:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError(
            "Aguarde a investigação atual terminar antes de excluir o incidente"
        )
    storage.delete_incident(workspace_id, incident_id)
    return RedirectResponse(
        f"/w/{workspace_id}?deleted=1",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/w/{workspace_id}/i/{incident_id}/comment")
async def post_comment(
    workspace_id: str,
    incident_id: str,
    background_tasks: BackgroundTasks,
    content: Annotated[str, Form()],
    author: Annotated[str, Form()] = "Equipe N2/N3",
) -> RedirectResponse:
    storage.ensure_incident_exists(workspace_id, incident_id)
    storage.append_feed(
        workspace_id,
        incident_id,
        author=author,
        author_type="HUMAN",
        content=content,
    )
    if mentions_agent(content):
        background_tasks.add_task(
            investigator.respond_to_mention,
            workspace_id,
            incident_id,
            content,
            author,
        )
    return RedirectResponse(
        f"/w/{workspace_id}/i/{incident_id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.post("/w/{workspace_id}/i/{incident_id}/investigate")
async def investigate_now(
    workspace_id: str, incident_id: str, background_tasks: BackgroundTasks
) -> RedirectResponse:
    storage.ensure_incident_exists(workspace_id, incident_id)
    storage.append_feed(
        workspace_id,
        incident_id,
        author="Agente Incidnet",
        author_type="AGENT",
        kind="SYSTEM",
        content="Investigação solicitada. A coleta de telemetria foi iniciada.",
    )
    logger.info(
        "Investigação solicitada",
        extra={"workspace_id": workspace_id, "incident_id": incident_id},
    )
    background_tasks.add_task(investigator.investigate, workspace_id, incident_id)
    return RedirectResponse(
        f"/w/{workspace_id}/i/{incident_id}?running=1",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


# ---------------------------------------------------------------------------
# API JSON para agentes
# ---------------------------------------------------------------------------
# As rotas HTML acima servem a interface humana. O bloco abaixo expõe a mesma
# funcionalidade em JSON, para que agentes automatizados possam operar o
# Incidnet de ponta a ponta. O contrato completo é descrito em
# GET /api/agent/guide.

ALLOWED_STATUSES = {"GOOD", "BAD", "INCONCLUSIVE", "RESOLVED"}


class WorkspaceCreate(BaseModel):
    name: str


class SettingsUpdate(BaseModel):
    prompt_base: str | None = None
    cron_schedule: str | None = None


class IncidentCreate(BaseModel):
    title: str
    affected_users: list[str] = Field(default_factory=list)
    description: str = ""


class IncidentDetailsUpdate(BaseModel):
    title: str
    description: str = ""


class CommentCreate(BaseModel):
    content: str
    author: str = "Agente externo"


class ResolveRequest(BaseModel):
    resolved_by: str = "Agente externo"


class AnalysisDiagnosisInput(BaseModel):
    user_id: str
    classification: Literal["GOOD", "BAD", "INCONCLUSIVE"]
    justification: str
    source: str = "EXTERNAL_AGENT"
    period: dict[str, str | None] = Field(default_factory=dict)
    queries: list[str] = Field(default_factory=list)
    rows: list[dict[str, str]] = Field(default_factory=list)
    summary: dict[str, int | str] = Field(default_factory=dict)
    scope: dict[str, int | str] = Field(default_factory=dict)


class AnalysisCreate(BaseModel):
    type: str = "agent_analysis"
    agent_name: str = "Agente externo"
    content: str = ""
    parent_analysis_id: str | None = None
    diagnoses: list[AnalysisDiagnosisInput] = Field(default_factory=list)


def _validate_cron(expression: str) -> None:
    from apscheduler.triggers.cron import CronTrigger

    try:
        CronTrigger.from_crontab(expression)
    except ValueError as exc:
        raise StorageError("Expressão cron inválida") from exc


def _diagnosis_payload(item) -> dict:
    data = item.model_dump(mode="json")
    data["status"] = item.status
    return data


EXPORT_COLUMNS = [
    "user_id",
    "classification",
    "status",
]


def _export_rows(diagnoses: list) -> list[dict]:
    rows = []
    for item in diagnoses:
        rows.append(
            {
                "user_id": item.user_id,
                "classification": item.classification,
                "status": item.status,
            }
        )
    return rows


def _incident_state(workspace_id: str, incident_id: str):
    incident = storage.get_incident(workspace_id, incident_id)
    diagnoses = storage.list_diagnoses(workspace_id, incident_id)
    counts = {
        key: sum(item.status == key for item in diagnoses)
        for key in ("GOOD", "BAD", "INCONCLUSIVE", "RESOLVED")
    }
    health_pct = (
        round((counts["GOOD"] + counts["RESOLVED"]) / len(incident.affected_users) * 100)
        if incident.affected_users
        else 0
    )
    return incident, diagnoses, counts, health_pct


@app.get("/api/workspaces")
async def api_list_workspaces() -> list[dict]:
    return [workspace.model_dump(mode="json") for workspace in storage.list_workspaces()]


@app.post("/api/workspaces", status_code=status.HTTP_201_CREATED)
async def api_create_workspace(payload: WorkspaceCreate) -> dict:
    workspace = storage.create_workspace(payload.name)
    scheduler.refresh()
    return workspace.model_dump(mode="json")


@app.get("/api/workspaces/{workspace_id}")
async def api_get_workspace(workspace_id: str) -> dict:
    workspace = storage.get_workspace(workspace_id)
    data = workspace.model_dump(mode="json")
    data["prompt_base"] = storage.get_prompt(workspace_id)
    data["incidents"] = [
        incident.model_dump(mode="json")
        for incident in storage.list_incidents(workspace_id)
    ]
    return data


@app.post("/api/workspaces/{workspace_id}/settings")
async def api_update_settings(workspace_id: str, payload: SettingsUpdate) -> dict:
    workspace = storage.get_workspace(workspace_id)
    cron_schedule = payload.cron_schedule or workspace.cron_schedule
    _validate_cron(cron_schedule)
    prompt = payload.prompt_base
    if prompt is None:
        prompt = storage.get_prompt(workspace_id)
    updated = storage.update_workspace(
        workspace_id,
        prompt=prompt,
        cron_schedule=cron_schedule,
    )
    scheduler.refresh()
    data = updated.model_dump(mode="json")
    data["prompt_base"] = storage.get_prompt(workspace_id)
    return data


@app.get("/api/workspaces/{workspace_id}/incidents")
async def api_list_incidents(workspace_id: str) -> list[dict]:
    storage.ensure_workspace_exists(workspace_id)
    return [
        incident.model_dump(mode="json")
        for incident in storage.list_incidents(workspace_id)
    ]


@app.post(
    "/api/workspaces/{workspace_id}/incidents",
    status_code=status.HTTP_201_CREATED,
)
async def api_create_incident(workspace_id: str, payload: IncidentCreate) -> dict:
    incident = storage.create_incident(
        workspace_id,
        payload.title,
        payload.affected_users,
        description=payload.description,
    )
    return incident.model_dump(mode="json")


@app.get("/api/workspaces/{workspace_id}/incidents/{incident_id}")
async def api_get_incident(workspace_id: str, incident_id: str) -> dict:
    storage.ensure_workspace_exists(workspace_id)
    incident, diagnoses, counts, health_pct = _incident_state(
        workspace_id, incident_id
    )
    feed = storage.get_feed(workspace_id, incident_id)
    return {
        "incident": incident.model_dump(mode="json"),
        "counts": counts,
        "health": health_pct,
        "running": investigator.is_running(workspace_id, incident_id),
        "diagnoses": [_diagnosis_payload(item) for item in diagnoses],
        "feed": [entry.model_dump(mode="json") for entry in feed.entries],
        "analyses": [
            item.model_dump(mode="json")
            for item in storage.list_analyses(workspace_id, incident_id)
        ],
    }


@app.get("/api/workspaces/{workspace_id}/incidents/{incident_id}/analyses")
async def api_list_analyses(workspace_id: str, incident_id: str) -> list[dict]:
    return [
        item.model_dump(mode="json")
        for item in storage.list_analyses(workspace_id, incident_id)
    ]


@app.get("/api/workspaces/{workspace_id}/incidents/{incident_id}/analyses/latest")
async def api_get_latest_analysis(workspace_id: str, incident_id: str) -> dict:
    analysis = storage.get_latest_analysis(workspace_id, incident_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Ainda não há análises para este incidente")
    return analysis.model_dump(mode="json")


@app.get("/api/workspaces/{workspace_id}/incidents/{incident_id}/analyses/{analysis_id}")
async def api_get_analysis(workspace_id: str, incident_id: str, analysis_id: str) -> dict:
    analysis = storage.get_analysis(workspace_id, incident_id, analysis_id)
    return {
        **analysis.model_dump(mode="json"),
        "diagnoses": [
            _diagnosis_payload(item)
            for item in storage.list_analysis_diagnoses(workspace_id, incident_id, analysis_id)
        ],
    }


@app.post(
    "/api/workspaces/{workspace_id}/incidents/{incident_id}/analyses",
    status_code=status.HTTP_201_CREATED,
)
async def api_create_analysis(
    workspace_id: str, incident_id: str, payload: AnalysisCreate
) -> dict:
    diagnoses = [
        UserDiagnosis(**item.model_dump(), investigated_at=storage.now())
        for item in payload.diagnoses
    ]
    analysis = storage.create_analysis(
        workspace_id,
        incident_id,
        diagnoses=diagnoses,
        analysis_type=payload.type,
        agent_name=payload.agent_name,
        content=payload.content,
        parent_analysis_id=payload.parent_analysis_id,
    )
    storage.append_feed(
        workspace_id,
        incident_id,
        author=analysis.agent_name,
        author_type="AGENT",
        kind="DIAGNOSIS",
        content=(
            f"Análise [`{analysis.analysis_id}`](/w/{workspace_id}/i/{incident_id}/analyses/{analysis.analysis_id}) "
            f"publicada por **{analysis.agent_name}** · {analysis.summary.total_users} usuários · "
            f"{analysis.summary.good} GOOD · {analysis.summary.bad} BAD · "
            f"{analysis.summary.inconclusive} INCONCLUSIVE"
        ),
    )
    return analysis.model_dump(mode="json")


@app.get("/api/workspaces/{workspace_id}/incidents/{incident_id}/progress")
async def api_incident_progress(workspace_id: str, incident_id: str) -> dict:
    """Lightweight polling endpoint for the incident page's live progress panel.

    Deliberately omits heavy fields (telemetry `rows`, feed) so it stays cheap
    to poll every ~1.5s even for incidents with hundreds/thousands of users.
    """
    incident, diagnoses, counts, health_pct = _incident_state(
        workspace_id, incident_id
    )
    return {
        "running": investigator.is_running(workspace_id, incident_id),
        "status": incident.status,
        "counts": counts,
        "health": health_pct,
        "total_users": len(incident.affected_users),
        "progress": investigator.progress(workspace_id, incident_id),
        "users": [
            {
                "user_id": item.user_id,
                "status": item.status,
                "justification": item.justification,
                "resolved": item.resolved,
            }
            for item in diagnoses
        ],
    }


@app.get("/api/workspaces/{workspace_id}/incidents/{incident_id}/diagnoses")
async def api_list_diagnoses(
    workspace_id: str, incident_id: str, status_name: str | None = None
) -> list[dict]:
    storage.ensure_workspace_exists(workspace_id)
    diagnoses = storage.list_diagnoses(workspace_id, incident_id)
    if status_name is not None:
        selected = status_name.upper()
        if selected not in ALLOWED_STATUSES:
            raise HTTPException(status_code=404, detail="Status não encontrado")
        diagnoses = [item for item in diagnoses if item.status == selected]
    return [_diagnosis_payload(item) for item in diagnoses]


@app.patch("/api/workspaces/{workspace_id}/incidents/{incident_id}")
async def api_update_incident(
    workspace_id: str, incident_id: str, payload: IncidentDetailsUpdate
) -> dict:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError(
            "Aguarde a investigação atual terminar antes de alterar o contexto"
        )
    incident = storage.update_incident_details(
        workspace_id, incident_id, title=payload.title, description=payload.description
    )
    return incident.model_dump(mode="json")


@app.delete("/api/workspaces/{workspace_id}/incidents/{incident_id}")
async def api_delete_incident(workspace_id: str, incident_id: str) -> dict:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError(
            "Aguarde a investigação atual terminar antes de excluir o incidente"
        )
    storage.delete_incident(workspace_id, incident_id)
    return {"deleted": True, "incident_id": incident_id}


@app.post("/api/workspaces/{workspace_id}/incidents/{incident_id}/investigate")
async def api_investigate(
    workspace_id: str, incident_id: str, background_tasks: BackgroundTasks
) -> dict:
    storage.ensure_incident_exists(workspace_id, incident_id)
    if investigator.is_running(workspace_id, incident_id):
        return {"status": "already_running", "incident_id": incident_id}
    storage.append_feed(
        workspace_id,
        incident_id,
        author="Agente Incidnet",
        author_type="AGENT",
        kind="SYSTEM",
        content="Investigação solicitada via API. A coleta de telemetria foi iniciada.",
    )
    background_tasks.add_task(investigator.investigate, workspace_id, incident_id)
    return {"status": "started", "incident_id": incident_id}


@app.post("/api/workspaces/{workspace_id}/incidents/{incident_id}/comment")
async def api_post_comment(
    workspace_id: str, incident_id: str, payload: CommentCreate,
    background_tasks: BackgroundTasks,
) -> dict:
    storage.ensure_incident_exists(workspace_id, incident_id)
    entry = storage.append_feed(
        workspace_id,
        incident_id,
        author=payload.author,
        author_type="HUMAN",
        content=payload.content,
    )
    if mentions_agent(payload.content):
        background_tasks.add_task(
            investigator.respond_to_mention,
            workspace_id,
            incident_id,
            payload.content,
            payload.author,
        )
    return entry.model_dump(mode="json")


@app.post(
    "/api/workspaces/{workspace_id}/incidents/{incident_id}/users/{user_id}/resolve"
)
async def api_resolve_user(
    workspace_id: str, incident_id: str, user_id: str, payload: ResolveRequest
) -> dict:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError(
            "Aguarde a investigação atual terminar antes de marcar como resolvido"
        )
    diagnosis = storage.mark_user_resolved(
        workspace_id, incident_id, user_id, resolved_by=payload.resolved_by
    )
    storage.append_feed(
        workspace_id,
        incident_id,
        author=payload.resolved_by,
        author_type="HUMAN",
        kind="SYSTEM",
        content=f"Usuário `{user_id}` marcado como **resolvido** por {payload.resolved_by}.",
    )
    return _diagnosis_payload(diagnosis)


@app.post(
    "/api/workspaces/{workspace_id}/incidents/{incident_id}/users/{user_id}/reopen"
)
async def api_reopen_user(
    workspace_id: str, incident_id: str, user_id: str
) -> dict:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError("Aguarde a investigação atual terminar antes de reabrir")
    diagnosis = storage.reopen_user(workspace_id, incident_id, user_id)
    storage.append_feed(
        workspace_id,
        incident_id,
        author="Agente Incidnet",
        author_type="AGENT",
        kind="SYSTEM",
        content=f"Usuário `{user_id}` reaberto para monitoramento.",
    )
    return _diagnosis_payload(diagnosis)


@app.delete(
    "/api/workspaces/{workspace_id}/incidents/{incident_id}/users/{user_id}"
)
async def api_delete_user(
    workspace_id: str, incident_id: str, user_id: str
) -> dict:
    if investigator.is_running(workspace_id, incident_id):
        raise StorageError(
            "Aguarde a investigação atual terminar antes de remover usuários"
        )
    incident = storage.remove_user(workspace_id, incident_id, user_id)
    storage.append_feed(
        workspace_id,
        incident_id,
        author="Agente Incidnet",
        author_type="AGENT",
        kind="SYSTEM",
        content=f"O usuário `{user_id}` foi removido do monitoramento.",
    )
    return incident.model_dump(mode="json")


def _build_agent_guide(base_url: str) -> str:
    base = base_url.rstrip("/")
    return f"""# Guia de integração do Incidnet para agentes

O Incidnet monitora a experiência de usuários de streaming durante incidentes.
Cada incidente reúne uma lista de `user_id`; o agente investiga a telemetria de
cada um e os classifica em `GOOD`, `BAD` ou `INCONCLUSIVE`. Um humano ou agente
pode marcar um usuário como `RESOLVED` quando o problema for tratado.

Base URL desta instância: `{base}`
Todas as rotas abaixo aceitam e retornam JSON (`Content-Type: application/json`).
Não há autenticação: proteja a rede/instância adequadamente. O contrato OpenAPI
completo está em `{base}/openapi.json` (UI interativa em `{base}/docs`).

## Conceitos

- **Workspace**: unidade de organização dos incidentes (por produto, stack,
  squad, o que fizer sentido para o seu time). `workspace_id` no formato
  `ws_xxxxxxxx`.
- **Incidente (incident)**: investigação com uma lista de `affected_users`.
  `incident_id` no formato `inc_xxxxxxxx`. Campo `status`: `OPEN`, `RESOLVED`
  (automático, quando todos os usuários monitorados estão `RESOLVED`) ou
  `CLOSED` (manual).
- **Diagnóstico (diagnosis)**: resultado por usuário. Campo `status` efetivo:
  - `GOOD` — há dados do device alvo e nenhuma evidência de falha.
  - `BAD` — há evidência de erro ou degradação.
  - `INCONCLUSIVE` — dados insuficientes para avaliar.
  - `RESOLVED` — marcado manualmente/por agente como tratado. Preserva a
    classificação original em `classification` e sobrevive a re-investigações.
- **Cron**: cada workspace tem `cron_schedule` (timezone America/Sao_Paulo) que
  dispara a investigação automática dos incidentes `OPEN`.
- **Análise (analysis)**: snapshot imutável de uma rodada ou de um agente
  especializado, identificado por `ana_xxxxxxxxxxxx`. Pode referenciar a análise
  que a originou por `parent_analysis_id`.

## Fluxo recomendado

1. Criar o workspace: `POST /api/workspaces`.
2. Ajustar regras de domínio e agenda: `POST /api/workspaces/{{workspace_id}}/settings`.
3. Abrir um incidente com os usuários afetados:
   `POST /api/workspaces/{{workspace_id}}/incidents`.
4. Disparar investigação sob demanda:
   `POST /api/workspaces/{{workspace_id}}/incidents/{{incident_id}}/investigate`.
5. Aguardar e ler o estado (a investigação roda em background — faça polling):
   `GET /api/workspaces/{{workspace_id}}/incidents/{{incident_id}}`.
6. Marcar usuários tratados como resolvidos:
   `POST .../incidents/{{incident_id}}/users/{{user_id}}/resolve`.

## Endpoints

### Workspaces
- `GET  /api/workspaces` — lista workspaces.
- `POST /api/workspaces` — cria workspace. Body: `{{"name": "Acme"}}`.
- `GET  /api/workspaces/{{workspace_id}}` — detalhes + `prompt_base` + incidentes.
- `POST /api/workspaces/{{workspace_id}}/settings` — atualiza regras e agenda.
  Body: `{{"prompt_base": "# regras...", "cron_schedule": "0 8 * * *"}}`
  (ambos opcionais; o que não for enviado é preservado).

### Incidentes
- `GET  /api/workspaces/{{workspace_id}}/incidents` — lista incidentes.
- `POST /api/workspaces/{{workspace_id}}/incidents` — cria incidente.
  Body: `{{"title": "Falha no Roku", "description": "contexto...",
  "affected_users": ["uuid-1", "uuid-2"]}}`.
- `GET  /api/workspaces/{{workspace_id}}/incidents/{{incident_id}}` — estado completo:
  `incident`, `counts` (por status), `health` (% GOOD+RESOLVED), `running`
  (investigação em andamento), `diagnoses` e `feed`.
- `PATCH /api/workspaces/{{workspace_id}}/incidents/{{incident_id}}` — edita
  título/descrição. Body: `{{"title": "...", "description": "..."}}`.
- `DELETE /api/workspaces/{{workspace_id}}/incidents/{{incident_id}}` — exclui.

### Investigação e diagnósticos
- `POST /api/workspaces/{{workspace_id}}/incidents/{{incident_id}}/investigate` —
  agenda uma nova rodada (background). Retorna `started` ou `already_running`.
- `GET  /api/workspaces/{{workspace_id}}/incidents/{{incident_id}}/diagnoses` —
  lista diagnósticos. Filtro opcional: `?status_name=BAD` (GOOD/BAD/INCONCLUSIVE/RESOLVED).
- `POST .../incidents/{{incident_id}}/comment` — adiciona comentário/instrução ao
  feed. Body: `{{"content": "texto", "author": "meu-agente"}}`. Se o texto
  mencionar `@agente` (ou `@agent`/`@incidnet`), o Agente Incidnet responde no
  feed com base na última rodada de diagnóstico concluída e na memória
  acumulada; dicas/correções relevantes são gravadas na memória do incidente.

### Análises versionadas
- `GET /api/workspaces/{{workspace_id}}/incidents/{{incident_id}}/analyses` —
  lista análises, da mais recente para a mais antiga.
- `GET .../analyses/latest` — recupera a última análise para outro agente
  continuar a investigação.
- `GET .../analyses/{{analysis_id}}` — análise completa, inclusive o snapshot de
  diagnósticos dela.
- `POST .../analyses` — publica uma análise externa. Body exemplo:
  `{{"type":"code_investigation", "agent_name":"Code Agent",
  "parent_analysis_id":"ana_...", "content":"Conclusão...",
  "diagnoses":[{{"user_id":"uuid-1", "classification":"BAD",
  "justification":"..."}}]}}`. Publicar não altera o estado operacional dos
  usuários; ele permanece separado para preservar o histórico.

### Memória persistente
- Comentários humanos ensinam o agente: dicas e correções viram memória e passam
  a orientar as próximas investigações. Mensagens do próprio agente (ex.:
  resultados de investigação) NÃO entram na memória, para poupar tokens.
- A memória existe em dois níveis: por incidente e por workspace (conhecimento
  geral, aplicável a incidentes futuros semelhantes). Uma rotina diária promove
  o que for reutilizável da memória dos incidentes para a memória do workspace.
- Toda investigação injeta a memória do workspace + do incidente no contexto do
  investigador quando há conteúdo.

### Usuários dentro do incidente
- `POST .../incidents/{{incident_id}}/users/{{user_id}}/resolve` — marca resolvido.
  Body: `{{"resolved_by": "meu-agente"}}`.
- `POST .../incidents/{{incident_id}}/users/{{user_id}}/reopen` — reabre.
- `DELETE .../incidents/{{incident_id}}/users/{{user_id}}` — remove o usuário.

## Boas práticas para agentes

- A investigação é assíncrona: após `investigate`, faça polling em `GET .../incidents/{{incident_id}}`
  até `running` ser `false`, então leia `diagnoses`/`counts`.
- Só marque `resolve` após confirmar que o usuário está `GOOD`/tratado; a marca
  é preservada mesmo se o incidente for re-investigado.
- Não crie workspaces/incidentes duplicados: consulte `GET /api/workspaces` antes.
- Enquanto `running` for `true`, operações de escrita no incidente (editar,
  excluir, resolver, remover usuário) são recusadas — tente novamente depois.
"""


@app.get("/api/agent/guide")
async def api_agent_guide(request: Request, format: str = "json"):
    guide = _build_agent_guide(str(request.base_url))
    if format == "md":
        return PlainTextResponse(guide, media_type="text/markdown; charset=utf-8")
    return {
        "version": app.version,
        "base_url": str(request.base_url).rstrip("/"),
        "openapi": str(request.base_url).rstrip("/") + "/openapi.json",
        "instructions": guide,
    }
