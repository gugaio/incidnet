from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .adapters import get_adapter, load_env_mcp_config
from .config import WORKSPACE_ROOT
from .log import get_logger
from .models import Analysis, AnalysisSummary, Feed, FeedEntry, Incident, Memory, MemoryEntry, UserDiagnosis, Workspace

logger = get_logger(__name__)


ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class StorageError(RuntimeError):
    pass


class NotFoundError(StorageError):
    pass


class FileStorage:
    """Small, atomic filesystem repository. All writes remain below workspace/."""

    def __init__(self, root: Path = WORKSPACE_ROOT) -> None:
        self.root = root.resolve()
        self.workspaces_root = self.root / "workspaces"
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_diagnoses()

    def _migrate_legacy_diagnoses(self) -> None:
        mapping = {
            "HEALTHY": "GOOD",
            "UNHEALTHY": "BAD",
            "INTERMITTENT": "INCONCLUSIVE",
            "ERROR": "INCONCLUSIVE",
        }
        for path in self.workspaces_root.glob(
            "*/incidents/*/users/*.json"
        ):
            try:
                data = self._read_json(path)
            except StorageError:
                logger.warning("Migração: erro ao ler diagnóstico", extra={"path": str(path)})
                continue
            current = data.get("classification")
            if current in mapping:
                data["classification"] = mapping[current]
                self._write_json(path, data)

    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def validate_id(value: str, label: str = "id") -> str:
        value = value.strip()
        if not ID_PATTERN.fullmatch(value):
            raise StorageError(f"{label} inválido")
        return value

    def _workspace_dir(self, workspace_id: str) -> Path:
        return self.workspaces_root / self.validate_id(workspace_id, "workspace_id")

    def _incident_dir(self, workspace_id: str, incident_id: str) -> Path:
        return (
            self._workspace_dir(workspace_id)
            / "incidents"
            / self.validate_id(incident_id, "incident_id")
        )

    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise NotFoundError(f"Arquivo não encontrado: {path.name}") from exc
        except json.JSONDecodeError as exc:
            raise StorageError(f"JSON inválido: {path.name}") from exc

    def _write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n"
        with self._lock:
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", dir=path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def create_workspace(
        self,
        name: str,
    ) -> Workspace:
        name = name.strip()
        if not name:
            raise StorageError("O nome do workspace é obrigatório")
        workspace_id = f"ws_{uuid4().hex[:8]}"
        workspace = Workspace(
            workspace_id=workspace_id,
            name=name,
            created_at=self.now(),
        )
        base = self._workspace_dir(workspace_id)
        (base / "incidents").mkdir(parents=True)
        self._write_json(base / "workspace.json", workspace.model_dump(mode="json"))
        self._write_json(base / "incidents" / "index.json", {"incidents": []})
        default_prompt = get_adapter(load_env_mcp_config()).default_prompt()
        (base / "PROMPT_BASE.md").write_text(default_prompt, encoding="utf-8")
        return workspace

    def list_workspaces(self) -> list[Workspace]:
        self.initialize()
        workspaces: list[Workspace] = []
        for path in sorted(self.workspaces_root.glob("*/workspace.json")):
            try:
                workspaces.append(Workspace.model_validate(self._read_json(path)))
            except (StorageError, ValueError):
                logger.warning("Workspace ignorado", extra={"path": str(path)})
                continue
        return sorted(workspaces, key=lambda item: item.created_at, reverse=True)

    def get_workspace(self, workspace_id: str) -> Workspace:
        path = self._workspace_dir(workspace_id) / "workspace.json"
        return Workspace.model_validate(self._read_json(path))

    def ensure_workspace_exists(self, workspace_id: str) -> None:
        """Raise NotFoundError if the workspace doesn't exist; discard the result.

        Use this (instead of `get_workspace`) when a route only needs to
        validate the workspace_id before acting, and has no use for the
        `Workspace` object itself.
        """
        self.get_workspace(workspace_id)

    def update_workspace(
        self,
        workspace_id: str,
        *,
        prompt: str,
        cron_schedule: str,
    ) -> Workspace:
        workspace = self.get_workspace(workspace_id)
        workspace.cron_schedule = cron_schedule.strip() or "0 0 * * *"
        base = self._workspace_dir(workspace_id)
        self._write_json(base / "workspace.json", workspace.model_dump(mode="json"))
        (base / "PROMPT_BASE.md").write_text(prompt, encoding="utf-8")
        return workspace

    def get_prompt(self, workspace_id: str) -> str:
        try:
            return (self._workspace_dir(workspace_id) / "PROMPT_BASE.md").read_text(
                encoding="utf-8"
            )
        except FileNotFoundError as exc:
            raise NotFoundError("PROMPT_BASE.md não encontrado") from exc

    def create_incident(
        self,
        workspace_id: str,
        title: str,
        user_ids: list[str],
        description: str = "",
    ) -> Incident:
        self.get_workspace(workspace_id)
        title = title.strip()
        if not title:
            raise StorageError("O título do incidente é obrigatório")
        normalized = list(
            dict.fromkeys(self.validate_id(item, "user_id") for item in user_ids if item)
        )
        if not normalized:
            raise StorageError("Informe ao menos um user_id")
        incident_id = f"inc_{uuid4().hex[:8]}"
        incident = Incident(
            incident_id=incident_id,
            title=title,
            description=description.strip(),
            created_at=self.now(),
            affected_users=normalized,
        )
        base = self._incident_dir(workspace_id, incident_id)
        (base / "users").mkdir(parents=True)
        self._write_json(base / "incident.json", incident.model_dump(mode="json"))
        feed = Feed(
            entries=[
                FeedEntry(
                    id=f"entry_{uuid4().hex[:10]}",
                    timestamp=self.now(),
                    author="Agente Incidnet",
                    author_type="AGENT",
                    kind="SYSTEM",
                    content=(
                        f"Incidente registrado e **{len(normalized)} usuários** "
                        "vinculados para monitoramento."
                    ),
                )
            ]
        )
        self._write_json(base / "feed.json", feed.model_dump(mode="json"))
        index_path = self._workspace_dir(workspace_id) / "incidents" / "index.json"
        with self._lock:
            index = self._read_json(index_path)
            index.setdefault("incidents", []).append(
                incident.model_dump(mode="json")
            )
            self._write_json(index_path, index)
        return incident

    def list_incidents(self, workspace_id: str) -> list[Incident]:
        path = self._workspace_dir(workspace_id) / "incidents" / "index.json"
        data = self._read_json(path)
        incidents = [Incident.model_validate(item) for item in data.get("incidents", [])]
        return sorted(incidents, key=lambda item: item.created_at, reverse=True)

    def get_incident(self, workspace_id: str, incident_id: str) -> Incident:
        path = self._incident_dir(workspace_id, incident_id) / "incident.json"
        return Incident.model_validate(self._read_json(path))

    def ensure_incident_exists(self, workspace_id: str, incident_id: str) -> None:
        """Raise NotFoundError if the incident doesn't exist; discard the result.

        Use this (instead of `get_incident`) when a route only needs to
        validate the workspace_id/incident_id before acting (e.g. before
        appending to the feed or scheduling a background task), and has no
        use for the `Incident` object itself.
        """
        self.get_incident(workspace_id, incident_id)

    def _replace_incident_in_index(
        self, workspace_id: str, incident: Incident
    ) -> None:
        index_path = self._workspace_dir(workspace_id) / "incidents" / "index.json"
        index = self._read_json(index_path)
        incidents = index.get("incidents", [])
        replaced = False
        for position, item in enumerate(incidents):
            if item.get("incident_id") == incident.incident_id:
                incidents[position] = incident.model_dump(mode="json")
                replaced = True
                break
        if not replaced:
            raise NotFoundError("Incidente não encontrado no índice")
        index["incidents"] = incidents
        self._write_json(index_path, index)

    def remove_user(
        self, workspace_id: str, incident_id: str, user_id: str
    ) -> Incident:
        user_id = self.validate_id(user_id, "user_id")
        with self._lock:
            incident = self.get_incident(workspace_id, incident_id)
            if user_id not in incident.affected_users:
                raise NotFoundError("Usuário não vinculado ao incidente")
            incident.affected_users = [
                current
                for current in incident.affected_users
                if current != user_id
            ]
            incident_path = (
                self._incident_dir(workspace_id, incident_id) / "incident.json"
            )
            self._write_json(
                incident_path, incident.model_dump(mode="json")
            )
            self._replace_incident_in_index(workspace_id, incident)
            diagnosis_path = (
                self._incident_dir(workspace_id, incident_id)
                / "users"
                / f"{user_id}.json"
            )
            diagnosis_path.unlink(missing_ok=True)
        self._sync_incident_status(workspace_id, incident_id)
        return self.get_incident(workspace_id, incident_id)

    def update_incident_details(
        self,
        workspace_id: str,
        incident_id: str,
        *,
        title: str,
        description: str,
    ) -> Incident:
        title = title.strip()
        if not title:
            raise StorageError("O título do incidente é obrigatório")
        with self._lock:
            incident = self.get_incident(workspace_id, incident_id)
            incident.title = title
            incident.description = description.strip()
            incident_path = (
                self._incident_dir(workspace_id, incident_id) / "incident.json"
            )
            self._write_json(
                incident_path, incident.model_dump(mode="json")
            )
            self._replace_incident_in_index(workspace_id, incident)
        return incident

    def delete_incident(self, workspace_id: str, incident_id: str) -> None:
        incident_id = self.validate_id(incident_id, "incident_id")
        incident_dir = self._incident_dir(workspace_id, incident_id)
        with self._lock:
            self.get_incident(workspace_id, incident_id)
            index_path = (
                self._workspace_dir(workspace_id) / "incidents" / "index.json"
            )
            index = self._read_json(index_path)
            original = index.get("incidents", [])
            remaining = [
                item
                for item in original
                if item.get("incident_id") != incident_id
            ]
            if len(remaining) == len(original):
                raise NotFoundError("Incidente não encontrado no índice")
            index["incidents"] = remaining
            self._write_json(index_path, index)
            shutil.rmtree(incident_dir)

    def get_feed(self, workspace_id: str, incident_id: str) -> Feed:
        path = self._incident_dir(workspace_id, incident_id) / "feed.json"
        return Feed.model_validate(self._read_json(path))

    def append_feed(
        self,
        workspace_id: str,
        incident_id: str,
        *,
        author: str,
        author_type: str,
        content: str,
        kind: str = "COMMENT",
    ) -> FeedEntry:
        content = content.strip()
        if not content:
            raise StorageError("O comentário não pode estar vazio")
        entry = FeedEntry(
            id=f"entry_{uuid4().hex[:10]}",
            timestamp=self.now(),
            author=author.strip() or "Anônimo",
            author_type=author_type,  # type: ignore[arg-type]
            content=content,
            kind=kind,
        )
        path = self._incident_dir(workspace_id, incident_id) / "feed.json"
        with self._lock:
            feed = self.get_feed(workspace_id, incident_id)
            feed.entries.append(entry)
            self._write_json(path, feed.model_dump(mode="json"))
        return entry

    def save_diagnosis(
        self, workspace_id: str, incident_id: str, diagnosis: UserDiagnosis
    ) -> None:
        safe_user_id = self.validate_id(diagnosis.user_id, "user_id")
        path = self._incident_dir(workspace_id, incident_id) / "users" / f"{safe_user_id}.json"
        data = diagnosis.model_dump(mode="json")
        with self._lock:
            if path.exists():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                    if existing.get("resolved"):
                        data["resolved"] = existing["resolved"]
                        data["resolved_at"] = existing.get("resolved_at")
                        data["resolved_by"] = existing.get("resolved_by")
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "Diagnóstico existente corrompido, sobrescrevendo",
                        extra={"user_id": diagnosis.user_id, "error": str(exc)},
                    )
            self._write_json(path, data)

    def _sync_incident_status(self, workspace_id: str, incident_id: str) -> None:
        """Auto-flip an incident between OPEN and RESOLVED.

        An incident becomes RESOLVED once every affected user has a resolved
        diagnosis, and reverts to OPEN if that stops being true (e.g. a user
        is reopened). Incidents manually set to CLOSED are left untouched.
        """
        with self._lock:
            incident = self.get_incident(workspace_id, incident_id)
            if incident.status == "CLOSED":
                return
            diagnoses_by_user = {
                item.user_id: item
                for item in self.list_diagnoses(workspace_id, incident_id)
            }
            all_resolved = bool(incident.affected_users) and all(
                diagnoses_by_user.get(user_id) is not None
                and diagnoses_by_user[user_id].resolved
                for user_id in incident.affected_users
            )
            new_status = "RESOLVED" if all_resolved else "OPEN"
            if new_status == incident.status:
                return
            incident.status = new_status
            incident_path = (
                self._incident_dir(workspace_id, incident_id) / "incident.json"
            )
            self._write_json(incident_path, incident.model_dump(mode="json"))
            self._replace_incident_in_index(workspace_id, incident)

    def mark_user_resolved(
        self,
        workspace_id: str,
        incident_id: str,
        user_id: str,
        *,
        resolved_by: str,
    ) -> UserDiagnosis:
        safe_user_id = self.validate_id(user_id, "user_id")
        path = self._incident_dir(workspace_id, incident_id) / "users" / f"{safe_user_id}.json"
        with self._lock:
            if not path.exists():
                raise NotFoundError(
                    f"Diagnóstico não encontrado para o usuário {user_id}"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            data["resolved"] = True
            data["resolved_at"] = self.now().isoformat()
            data["resolved_by"] = resolved_by.strip() or "Anônimo"
            self._write_json(path, data)
        self._sync_incident_status(workspace_id, incident_id)
        return UserDiagnosis.model_validate(data)

    def reopen_user(
        self,
        workspace_id: str,
        incident_id: str,
        user_id: str,
    ) -> UserDiagnosis:
        safe_user_id = self.validate_id(user_id, "user_id")
        path = self._incident_dir(workspace_id, incident_id) / "users" / f"{safe_user_id}.json"
        with self._lock:
            if not path.exists():
                raise NotFoundError(
                    f"Diagnóstico não encontrado para o usuário {user_id}"
                )
            data = json.loads(path.read_text(encoding="utf-8"))
            data["resolved"] = False
            data["resolved_at"] = None
            data["resolved_by"] = None
            self._write_json(path, data)
        self._sync_incident_status(workspace_id, incident_id)
        return UserDiagnosis.model_validate(data)

    def list_diagnoses(
        self, workspace_id: str, incident_id: str
    ) -> list[UserDiagnosis]:
        users_dir = self._incident_dir(workspace_id, incident_id) / "users"
        diagnoses = []
        for path in sorted(users_dir.glob("*.json")):
            try:
                diagnoses.append(UserDiagnosis.model_validate(self._read_json(path)))
            except (StorageError, ValueError):
                logger.warning("Diagnóstico inválido ignorado", extra={"path": str(path)})
                continue
        return diagnoses

    # ------------------------------------------------------------------
    # Immutable analysis rounds
    # ------------------------------------------------------------------

    def _analyses_dir(self, workspace_id: str, incident_id: str) -> Path:
        return self._incident_dir(workspace_id, incident_id) / "analyses"

    def _analysis_dir(
        self, workspace_id: str, incident_id: str, analysis_id: str
    ) -> Path:
        return self._analyses_dir(workspace_id, incident_id) / self.validate_id(
            analysis_id, "analysis_id"
        )

    def create_analysis(
        self,
        workspace_id: str,
        incident_id: str,
        *,
        diagnoses: list[UserDiagnosis],
        analysis_type: str = "telemetry_diagnosis",
        agent_name: str = "Agente Incidnet",
        content: str = "",
        parent_analysis_id: str | None = None,
    ) -> Analysis:
        incident = self.get_incident(workspace_id, incident_id)
        if parent_analysis_id is not None:
            self.get_analysis(workspace_id, incident_id, parent_analysis_id)
        summary = AnalysisSummary(
            total_users=len(diagnoses),
            good=sum(item.classification == "GOOD" for item in diagnoses),
            bad=sum(item.classification == "BAD" for item in diagnoses),
            inconclusive=sum(item.classification == "INCONCLUSIVE" for item in diagnoses),
        )
        analysis = Analysis(
            analysis_id=f"ana_{uuid4().hex[:12]}",
            incident_id=incident.incident_id,
            created_at=self.now(),
            type=analysis_type.strip() or "agent_analysis",
            agent_name=agent_name.strip() or "Agente Incidnet",
            parent_analysis_id=parent_analysis_id,
            content=content.strip(),
            summary=summary,
        )
        base = self._analysis_dir(workspace_id, incident_id, analysis.analysis_id)
        with self._lock:
            self._write_json(base / "analysis.json", analysis.model_dump(mode="json"))
            for diagnosis in diagnoses:
                user_id = self.validate_id(diagnosis.user_id, "user_id")
                self._write_json(
                    base / "users" / f"{user_id}.json",
                    diagnosis.model_dump(mode="json"),
                )
        return analysis

    def list_analyses(self, workspace_id: str, incident_id: str) -> list[Analysis]:
        self.ensure_incident_exists(workspace_id, incident_id)
        analyses: list[Analysis] = []
        for path in self._analyses_dir(workspace_id, incident_id).glob("*/analysis.json"):
            try:
                analyses.append(Analysis.model_validate(self._read_json(path)))
            except (StorageError, ValueError):
                logger.warning("Análise inválida ignorada", extra={"path": str(path)})
        return sorted(analyses, key=lambda item: item.created_at, reverse=True)

    def get_analysis(
        self, workspace_id: str, incident_id: str, analysis_id: str
    ) -> Analysis:
        return Analysis.model_validate(
            self._read_json(self._analysis_dir(workspace_id, incident_id, analysis_id) / "analysis.json")
        )

    def get_latest_analysis(self, workspace_id: str, incident_id: str) -> Analysis | None:
        analyses = self.list_analyses(workspace_id, incident_id)
        return analyses[0] if analyses else None

    def list_analysis_diagnoses(
        self, workspace_id: str, incident_id: str, analysis_id: str
    ) -> list[UserDiagnosis]:
        base = self._analysis_dir(workspace_id, incident_id, analysis_id) / "users"
        diagnoses: list[UserDiagnosis] = []
        for path in sorted(base.glob("*.json")):
            try:
                diagnoses.append(UserDiagnosis.model_validate(self._read_json(path)))
            except (StorageError, ValueError):
                logger.warning("Diagnóstico de análise inválido ignorado", extra={"path": str(path)})
        return diagnoses

    # ------------------------------------------------------------------
    # Persistent memory
    # ------------------------------------------------------------------
    # Durable knowledge taught by humans through the feed. Only the incident and
    # workspace memories feed future investigations — agent-authored feed entries
    # (e.g. diagnosis rounds) are deliberately excluded to save tokens.

    @staticmethod
    def _normalize_memory(content: str) -> str:
        return " ".join(content.split()).strip().lower()

    def _incident_memory_path(self, workspace_id: str, incident_id: str) -> Path:
        return self._incident_dir(workspace_id, incident_id) / "memory.json"

    def _workspace_memory_path(self, workspace_id: str) -> Path:
        return self._workspace_dir(workspace_id) / "memory.json"

    def _read_memory(self, path: Path) -> Memory:
        try:
            return Memory.model_validate(self._read_json(path))
        except (NotFoundError, StorageError, ValueError):
            return Memory()

    def get_incident_memory(self, workspace_id: str, incident_id: str) -> Memory:
        return self._read_memory(
            self._incident_memory_path(workspace_id, incident_id)
        )

    def get_workspace_memory(self, workspace_id: str) -> Memory:
        return self._read_memory(self._workspace_memory_path(workspace_id))

    def append_incident_memory(
        self,
        workspace_id: str,
        incident_id: str,
        content: str,
        *,
        author: str = "Agente Incidnet",
    ) -> MemoryEntry | None:
        content = content.strip()
        if not content:
            return None
        path = self._incident_memory_path(workspace_id, incident_id)
        with self._lock:
            memory = self._read_memory(path)
            normalized = self._normalize_memory(content)
            if any(
                self._normalize_memory(entry.content) == normalized
                for entry in memory.entries
            ):
                return None
            entry = MemoryEntry(
                id=f"mem_{uuid4().hex[:10]}",
                timestamp=self.now(),
                content=content,
                author=author.strip() or "Agente Incidnet",
                source_incident_id=incident_id,
            )
            memory.entries.append(entry)
            self._write_json(path, memory.model_dump(mode="json"))
        return entry

    def append_workspace_memory(
        self,
        workspace_id: str,
        content: str,
        *,
        author: str = "Agente Incidnet",
        source_incident_id: str | None = None,
    ) -> MemoryEntry | None:
        content = content.strip()
        if not content:
            return None
        path = self._workspace_memory_path(workspace_id)
        with self._lock:
            memory = self._read_memory(path)
            normalized = self._normalize_memory(content)
            if any(
                self._normalize_memory(entry.content) == normalized
                for entry in memory.entries
            ):
                return None
            entry = MemoryEntry(
                id=f"mem_{uuid4().hex[:10]}",
                timestamp=self.now(),
                content=content,
                author=author.strip() or "Agente Incidnet",
                source_incident_id=source_incident_id,
            )
            memory.entries.append(entry)
            self._write_json(path, memory.model_dump(mode="json"))
        return entry

    def build_memory_context(self, workspace_id: str, incident_id: str) -> str:
        """Render workspace + incident memory as a compact text block for the LLM.

        Returns an empty string when there is nothing persisted yet.
        """
        workspace_memory = self.get_workspace_memory(workspace_id)
        incident_memory = self.get_incident_memory(workspace_id, incident_id)
        sections: list[str] = []
        if workspace_memory.entries:
            lines = "\n".join(
                f"- {entry.content}" for entry in workspace_memory.entries
            )
            sections.append(
                "Conhecimento acumulado do workspace (aplicável a incidentes "
                f"semelhantes):\n{lines}"
            )
        if incident_memory.entries:
            lines = "\n".join(
                f"- {entry.content}" for entry in incident_memory.entries
            )
            sections.append(
                f"Conhecimento específico deste incidente:\n{lines}"
            )
        return "\n\n".join(sections)
