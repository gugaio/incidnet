from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

from .adapters import MCPConfigurationError, get_adapter, load_env_mcp_config, redact_exception
from .adapters.base import MCPAdapter
from .config import INVESTIGATION_CONCURRENCY
from .llm import (
    analyze_telemetry,
    answer_mention,
    consolidate_workspace_memory,
    extract_memory_note,
)
from .log import get_logger
from .models import UserDiagnosis
from .storage import FileStorage

logger = get_logger(__name__)


class Investigator:
    def __init__(self, storage: FileStorage) -> None:
        self.storage = storage
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._progress: dict[tuple[str, str], dict[str, Any]] = {}

    def _lock_for(self, workspace_id: str, incident_id: str) -> asyncio.Lock:
        return self._locks.setdefault((workspace_id, incident_id), asyncio.Lock())

    def is_running(self, workspace_id: str, incident_id: str) -> bool:
        return self._lock_for(workspace_id, incident_id).locked()

    def progress(self, workspace_id: str, incident_id: str) -> dict[str, Any] | None:
        """Live progress for the current/last investigation round, if any."""
        state = self._progress.get((workspace_id, incident_id))
        if state is None:
            return None
        return {
            **state,
            "counts": dict(state["counts"]),
            "in_flight": list(state["in_flight"]),
        }

    def _reset_progress(self, key: tuple[str, str], total: int) -> None:
        self._progress[key] = {
            "phase": "connecting",
            "total": total,
            "done": 0,
            "counts": {},
            "in_flight": [],
            "message": "Conectando ao MCP…",
            "started_at": self.storage.now().isoformat(),
            "finished_at": None,
        }

    def _update_progress(self, key: tuple[str, str], **changes: Any) -> None:
        state = self._progress.get(key)
        if state is None:
            return
        state.update(changes)

    async def investigate(self, workspace_id: str, incident_id: str) -> None:
        lock = self._lock_for(workspace_id, incident_id)
        if lock.locked():
            self.storage.append_feed(
                workspace_id,
                incident_id,
                author="Agente Incidnet",
                author_type="AGENT",
                content="Uma rodada de investigação já está em andamento.",
                kind="WARNING",
            )
            logger.warning(
                "Investigação já em andamento",
                extra={"workspace_id": workspace_id, "incident_id": incident_id},
            )
            return
        key = (workspace_id, incident_id)
        async with lock:
            incident = self.storage.get_incident(workspace_id, incident_id)
            prompt = self.storage.get_prompt(workspace_id)
            memory = self.storage.build_memory_context(workspace_id, incident_id)
            adapter = get_adapter(load_env_mcp_config())
            token = adapter.secret()
            users = incident.affected_users
            self._reset_progress(key, total=len(users))
            results: list[UserDiagnosis] = []
            logger.info(
                "Investigação iniciada",
                extra={
                    "workspace_id": workspace_id,
                    "incident_id": incident_id,
                    "total_users": len(users),
                    "incident_title": incident.title,
                },
            )
            try:
                async with adapter.session() as session:
                    logger.debug(
                        "Sessão MCP estabelecida",
                        extra={"workspace_id": workspace_id, "incident_id": incident_id},
                    )
                    tools = await session.list_tools()
                    names = {tool.name for tool in tools.tools}
                    missing = adapter.required_tools() - names
                    if missing:
                        raise MCPConfigurationError(
                            "O MCP não expõe a(s) tool(s) obrigatória(s): "
                            + ", ".join(sorted(missing))
                        )
                    self._update_progress(key, phase="running", message="")
                    semaphore = asyncio.Semaphore(INVESTIGATION_CONCURRENCY)

                    async def investigate_one(user_id: str) -> None:
                        async with semaphore:
                            state = self._progress.get(key)
                            if state is not None:
                                state["in_flight"].append(user_id)
                            try:
                                diagnosis = await self._investigate_user(
                                    adapter,
                                    session,
                                    user_id,
                                    prompt,
                                    incident.title,
                                    incident.description,
                                    memory,
                                )
                                logger.info(
                                    "Usuário investigado",
                                    extra={
                                        "user_id": user_id,
                                        "classification": diagnosis.classification,
                                        "source": diagnosis.source,
                                    },
                                )
                            except Exception as exc:
                                logger.error(
                                    "Falha ao investigar usuário",
                                    extra={"user_id": user_id, "error": redact_exception(exc, (token,))},
                                )
                                diagnosis = UserDiagnosis(
                                    user_id=user_id,
                                    classification="INCONCLUSIVE",
                                    justification=(
                                        "Falha ao consultar este usuário: "
                                        + redact_exception(exc, (token,))
                                    ),
                                    investigated_at=self.storage.now(),
                                    source="MCP_ERROR",
                                )
                            self.storage.save_diagnosis(
                                workspace_id, incident_id, diagnosis
                            )
                            results.append(diagnosis)
                            state = self._progress.get(key)
                            if state is not None:
                                state["done"] += 1
                                state["counts"][diagnosis.classification] = (
                                    state["counts"].get(diagnosis.classification, 0)
                                    + 1
                                )
                                try:
                                    state["in_flight"].remove(user_id)
                                except ValueError:
                                    pass

                    await asyncio.gather(
                        *(investigate_one(user_id) for user_id in users)
                    )
            except Exception as exc:
                message = redact_exception(exc, (token,))
                logger.error(
                    "Investigação falhou",
                    extra={
                        "workspace_id": workspace_id,
                        "incident_id": incident_id,
                        "error": message,
                    },
                )
                self._update_progress(
                    key,
                    phase="error",
                    message=message,
                    finished_at=self.storage.now().isoformat(),
                )
                self.storage.append_feed(
                    workspace_id,
                    incident_id,
                    author="Agente Incidnet",
                    author_type="AGENT",
                    kind="ERROR",
                    content=(
                        "### Falha na rodada de investigação\n\n"
                        f"O servidor MCP está indisponível ou rejeitou a consulta: "
                        f"`{message}`\n\nNenhum dado foi descartado."
                    ),
                )
                return
            self._update_progress(
                key, phase="done", finished_at=self.storage.now().isoformat()
            )
            results_by_user = {item.user_id: item for item in results}
            ordered_results = [
                results_by_user[user_id]
                for user_id in users
                if user_id in results_by_user
            ]
            self._post_summary(workspace_id, incident_id, ordered_results)
            counts = Counter(r.classification for r in ordered_results)
            logger.info(
                "Investigação concluída",
                extra={
                    "workspace_id": workspace_id,
                    "incident_id": incident_id,
                    "total": len(ordered_results),
                    "good": counts["GOOD"],
                    "bad": counts["BAD"],
                    "inconclusive": counts["INCONCLUSIVE"],
                },
            )

    async def _investigate_user(
        self,
        adapter: MCPAdapter,
        session: Any,
        user_id: str,
        prompt: str,
        incident_title: str,
        incident_description: str,
        memory: str = "",
    ) -> UserDiagnosis:
        telemetry = await adapter.query_user(
            session,
            user_id,
            incident_title=incident_title,
            incident_description=incident_description,
        )
        analysis, source = await analyze_telemetry(
            user_id=user_id,
            incident_title=incident_title,
            incident_description=incident_description,
            prompt_base=prompt,
            telemetry=telemetry,
            classify=adapter.classify,
            memory=memory,
        )
        return UserDiagnosis(
            user_id=user_id,
            classification=analysis.classification,
            justification=analysis.justification,
            investigated_at=self.storage.now(),
            source=source,
            period=telemetry["period"],
            queries=telemetry["queries"],
            rows=telemetry["rows"],
            summary=telemetry["summary"],
            scope=telemetry["scope"],
        )

    def _post_summary(
        self,
        workspace_id: str,
        incident_id: str,
        results: list[UserDiagnosis],
    ) -> None:
        analysis = self.storage.create_analysis(
            workspace_id, incident_id, diagnoses=results,
            analysis_type="telemetry_diagnosis", agent_name="Agente Incidnet",
        )
        summary = analysis.summary
        self.storage.append_feed(
            workspace_id,
            incident_id,
            author="Agente Incidnet",
            author_type="AGENT",
            kind="DIAGNOSIS",
            content=(
                "### Análise concluída\n\n"
                f"[`{analysis.analysis_id}`](/w/{workspace_id}/i/{incident_id}/analyses/{analysis.analysis_id}) "
                f"· **{summary.total_users} usuários** · "
                f"**{summary.good} GOOD** · **{summary.bad} BAD** · "
                f"**{summary.inconclusive} INCONCLUSIVE**"
            ),
        )

    async def investigate_all_open(self) -> None:
        logger.info("Iniciando investigação programada de todos os incidentes abertos")
        for workspace in self.storage.list_workspaces():
            for incident in self.storage.list_incidents(workspace.workspace_id):
                if incident.status == "OPEN":
                    await self.investigate(workspace.workspace_id, incident.incident_id)
        logger.info("Investigação programada concluída")

    def _build_mention_context(self, workspace_id: str, incident_id: str) -> str:
        """Build a rich, compact context for @mention responses.

        Combines the last diagnosis round feed summary with per-user data
        (status, scope, summary metrics, justification — but not raw rows so
        we keep the token count manageable).
        """
        feed = self.storage.get_feed(workspace_id, incident_id)
        last_round = ""
        for entry in reversed(feed.entries):
            if entry.kind == "DIAGNOSIS":
                last_round = entry.content
                break

        diagnoses = self.storage.list_diagnoses(workspace_id, incident_id)
        if not diagnoses:
            return last_round or "Ainda não há rodada de diagnóstico concluída."

        user_lines: list[str] = []
        for d in diagnoses:
            parts: list[str] = [f"{d.user_id}: {d.status}"]
            if d.scope:
                device = d.scope.get("target_device", "")
                matched = d.scope.get("matched_rows", "")
                total = d.scope.get("total_rows", "")
                if device:
                    parts.append(f"device={device}, rows={matched}/{total}")
            if d.summary:
                parts.append(
                    ", ".join(f"{k}={v}" for k, v in d.summary.items())
                )
            if d.period and d.period.get("from"):
                parts.append(f"período={d.period['from']} → {d.period.get('to', '')}")
            parts.append(d.justification)
            user_lines.append(" | ".join(parts))

        sections: list[str] = []
        if last_round:
            sections.append(last_round)
        sections.append(
            "**Diagnósticos por usuário (última rodada):**\n"
            + "\n".join(f"- {line}" for line in user_lines)
        )
        return "\n\n".join(sections)

    async def respond_to_mention(
        self,
        workspace_id: str,
        incident_id: str,
        question: str,
        author: str,
    ) -> None:
        """React to an @agente mention: reply in the feed and learn if relevant.

        Opens an MCP session so the LLM can issue arbitrary NQL queries when
        the question goes beyond what the last investigation round captured.
        Falls back gracefully when MCP is not configured or the session fails.
        """
        try:
            incident = self.storage.get_incident(workspace_id, incident_id)
            prompt = self.storage.get_prompt(workspace_id)
        except Exception:
            logger.exception(
                "Erro ao carregar dados para @mention",
                extra={"workspace_id": workspace_id, "incident_id": incident_id},
            )
            return

        context = self._build_mention_context(workspace_id, incident_id)
        memory = self.storage.build_memory_context(workspace_id, incident_id)

        logger.info(
            "@mention recebida",
            extra={
                "workspace_id": workspace_id,
                "incident_id": incident_id,
                "author": author,
            },
        )

        try:
            adapter = get_adapter(load_env_mcp_config())
            if adapter.validate_config():
                async with adapter.session() as session:
                    token = adapter.secret()

                    async def _mcp_query(nql: str, timeout: int = 60) -> str:
                        from .adapters.npaw import parse_query_result

                        try:
                            result = await session.call_tool(
                                "npaw_query_data",
                                arguments={"nql": nql, "timeout": timeout},
                            )
                            rows = parse_query_result(result)
                            if not rows:
                                return "Nenhuma row retornada para esta query."
                            return json.dumps(rows, ensure_ascii=False, default=str)
                        except Exception as exc:
                            from .adapters import redact_exception
                            return f"Erro na query: {redact_exception(exc, (token,))}"

                    answer = await answer_mention(
                        question=question,
                        incident_title=incident.title,
                        incident_description=incident.description,
                        prompt_base=prompt,
                        last_round_summary=context,
                        memory=memory,
                        mcp_query=_mcp_query,
                    )
            else:
                answer = await answer_mention(
                    question=question,
                    incident_title=incident.title,
                    incident_description=incident.description,
                    prompt_base=prompt,
                    last_round_summary=context,
                    memory=memory,
                    mcp_query=None,
                )
        except Exception as exc:
            logger.exception(
                "Erro no @mention (fallback sem MCP)",
                extra={"workspace_id": workspace_id, "incident_id": incident_id},
            )
            from .adapters import redact_exception
            token = ""
            try:
                token = get_adapter(load_env_mcp_config()).secret()
            except Exception:
                logger.warning("Não foi possível obter token para redação")
            err = redact_exception(exc, (token,))
            answer = await answer_mention(
                question=question,
                incident_title=incident.title,
                incident_description=incident.description,
                prompt_base=prompt,
                last_round_summary=context,
                memory=memory,
                mcp_query=None,
            )
            answer = f"⚠️ Erro ao abrir sessão MCP (`{err}`).\n\n" + answer

        self.storage.append_feed(
            workspace_id,
            incident_id,
            author="Agente Incidnet",
            author_type="AGENT",
            kind="COMMENT",
            content=answer,
        )

        note = await extract_memory_note(
            question=question,
            incident_title=incident.title,
            incident_description=incident.description,
        )
        if note:
            entry = self.storage.append_incident_memory(
                workspace_id,
                incident_id,
                note,
                author=author,
            )
            if entry is not None:
                self.storage.append_feed(
                    workspace_id,
                    incident_id,
                    author="Agente Incidnet",
                    author_type="AGENT",
                    kind="SYSTEM",
                    content=(
                        "📝 Aprendizado registrado na memória deste incidente e "
                        "será considerado nas próximas investigações:\n\n"
                        f"> {note}"
                    ),
                )
                logger.info(
                    "Memória do incidente atualizada via @mention",
                    extra={
                        "workspace_id": workspace_id,
                        "incident_id": incident_id,
                        "author": author,
                    },
                )

    async def consolidate_memories(self) -> None:
        """Daily routine: promote reusable incident knowledge to the workspace.

        Looks at every incident's memory and asks the LLM which notes deserve to
        live at the workspace level (i.e. are general enough to help future,
        similar incidents). Persists only genuinely new, generalized knowledge.
        """
        logger.info("Iniciando consolidação diária de memórias")
        for workspace in self.storage.list_workspaces():
            workspace_id = workspace.workspace_id
            incident_notes: list[str] = []
            for incident in self.storage.list_incidents(workspace_id):
                incident_memory = self.storage.get_incident_memory(
                    workspace_id, incident.incident_id
                )
                incident_notes.extend(
                    entry.content for entry in incident_memory.entries
                )
            if not incident_notes:
                continue
            existing = [
                entry.content
                for entry in self.storage.get_workspace_memory(workspace_id).entries
            ]
            promoted = await consolidate_workspace_memory(
                incident_notes=incident_notes,
                existing_workspace_memory=existing,
            )
            for note in promoted:
                self.storage.append_workspace_memory(
                    workspace_id,
                    note,
                    author="Agente Incidnet (consolidação diária)",
                )
            if promoted:
                logger.info(
                    "Memórias promovidas para o workspace",
                    extra={
                        "workspace_id": workspace_id,
                        "promoted": len(promoted),
                        "incident_notes": len(incident_notes),
                    },
                )
        logger.info("Consolidação diária de memórias concluída")
