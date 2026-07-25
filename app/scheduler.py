from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import DEFAULT_TIMEZONE
from .investigator import Investigator
from .log import get_logger
from .storage import FileStorage

logger = get_logger(__name__)


class InvestigationScheduler:
    def __init__(self, storage: FileStorage, investigator: Investigator) -> None:
        self.storage = storage
        self.investigator = investigator
        self.scheduler = AsyncIOScheduler(timezone=DEFAULT_TIMEZONE)

    def start(self) -> None:
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Scheduler iniciado")
        self.refresh()

    def refresh(self) -> None:
        existing = {job.id for job in self.scheduler.get_jobs()}
        desired: set[str] = set()
        self._ensure_memory_job(existing, desired)
        for workspace in self.storage.list_workspaces():
            job_id = f"workspace:{workspace.workspace_id}"
            desired.add(job_id)
            try:
                trigger = CronTrigger.from_crontab(
                    workspace.cron_schedule, timezone=DEFAULT_TIMEZONE
                )
            except ValueError:
                logger.warning(
                    "Cron inválido para workspace, usando padrão",
                    extra={
                        "workspace_id": workspace.workspace_id,
                        "cron": workspace.cron_schedule,
                    },
                )
                trigger = CronTrigger.from_crontab(
                    "0 0 * * *", timezone=DEFAULT_TIMEZONE
                )
            self.scheduler.add_job(
                self._run_workspace,
                trigger,
                id=job_id,
                args=[workspace.workspace_id],
                replace_existing=True,
                coalesce=True,
                max_instances=1,
            )
        for obsolete in existing - desired:
            self.scheduler.remove_job(obsolete)
        if desired - existing:
            logger.info("Agenda atualizada", extra={"jobs": len(desired)})

    def _ensure_memory_job(self, existing: set[str], desired: set[str]) -> None:
        """Register the daily memory-consolidation job (runs once, workspace-wide)."""
        job_id = "memory-consolidation"
        desired.add(job_id)
        self.scheduler.add_job(
            self._run_memory_consolidation,
            CronTrigger.from_crontab("0 4 * * *", timezone=DEFAULT_TIMEZONE),
            id=job_id,
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )

    async def _run_memory_consolidation(self) -> None:
        await self.investigator.consolidate_memories()

    async def _run_workspace(self, workspace_id: str) -> None:
        logger.info("Job disparado para workspace", extra={"workspace_id": workspace_id})
        for incident in self.storage.list_incidents(workspace_id):
            if incident.status == "OPEN":
                await self.investigator.investigate(
                    workspace_id, incident.incident_id
                )

    def shutdown(self) -> None:
        if self.scheduler.running:
            logger.info("Scheduler encerrando")
            self.scheduler.shutdown(wait=False)

