"""jobs 域的仓储类（Phase 1a 自 repositories.py 拆出，内容未改动）。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
import re
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy import Integer, String, and_, bindparam, case, cast, func, or_, select, text, true
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.storage.models import (
    BbotListenerCacheEntry,
    ConversationEpisode,
    DevSession,
    DevTask,
    DevTaskArtifact,
    EpisodeMessage,
    Group,
    GroupPersonaState,
    Job,
    MemoryBackfillRun,
    MemoryItem,
    MemoryItemSemanticVector,
    Message,
    PersonaStyleExample,
    PersonaStyleSyncState,
    RetrievalDocument,
    RetrievalDocumentMessage,
    RetrievalIndexState,
    Summary,
    UsageRecord,
    User,
)
from app.providers.embeddings import hashed_text_embedding
from app.storage.db import validate_retrieval_vector_table_name
from app.core.time_utils import shanghai_naive, shanghai_now_naive

from app.storage.repositories._common import _normalize_utc_sqlite_timestamp

class JobRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_job(
        self,
        *,
        job_type: str,
        payload_json: dict[str, Any],
        run_at: datetime,
        status: str = "queued",
        job_key: str = "",
    ) -> Job:
        if job_key:
            self.session.execute(
                text(
                    "INSERT OR IGNORE INTO jobs (job_type, job_key, payload_json, status, run_at) "
                    "VALUES (:job_type, :job_key, :payload_json, :status, :run_at)"
                ),
                {
                    "job_type": job_type,
                    "job_key": job_key,
                    "payload_json": json.dumps(payload_json),
                    "status": status,
                    "run_at": _normalize_utc_sqlite_timestamp(run_at),
                },
            )
            existing = self.session.scalars(
                select(Job).where(Job.job_type == job_type, Job.job_key == job_key)
            ).first()
            if existing is not None:
                return existing
        job = Job(job_type=job_type, job_key=job_key, payload_json=payload_json, run_at=run_at, status=status)
        self.session.add(job)
        self.session.flush()
        return job

    def enqueue_coalescing_job(
        self,
        *,
        job_type: str,
        job_key: str,
        payload_json: dict[str, Any],
        run_at: datetime,
        backfill_run_id: int | None = None,
        target_generation: str = "",
        max_attempts: int = 3,
    ) -> Job:
        if not job_key:
            raise ValueError("coalescing jobs require a stable job_key")
        job_id = self.session.execute(
            text(
                "INSERT INTO jobs ("
                "job_type, job_key, payload_json, status, run_at, "
                "requested_generation, processed_generation, claimed_generation, "
                "backfill_run_id, target_generation, max_attempts"
                ") VALUES ("
                ":job_type, :job_key, :payload_json, 'queued', :run_at, 1, 0, 0, "
                ":backfill_run_id, :target_generation, :max_attempts"
                ") ON CONFLICT(job_type, job_key) WHERE job_key <> '' DO UPDATE SET "
                "requested_generation = jobs.requested_generation + 1, "
                "payload_json = excluded.payload_json, "
                "backfill_run_id = coalesce("
                "excluded.backfill_run_id, jobs.backfill_run_id), "
                "target_generation = CASE WHEN excluded.target_generation <> '' "
                "THEN excluded.target_generation ELSE jobs.target_generation END, "
                "max_attempts = excluded.max_attempts, "
                "status = CASE WHEN jobs.status = 'running' THEN 'running' ELSE 'queued' END, "
                "run_at = CASE WHEN jobs.status = 'running' THEN jobs.run_at ELSE excluded.run_at END, "
                "completed_at = NULL, last_error_code = '' "
                "RETURNING id"
            ),
            {
                "job_type": job_type,
                "job_key": job_key,
                "payload_json": json.dumps(payload_json),
                "run_at": _normalize_utc_sqlite_timestamp(run_at),
                "backfill_run_id": backfill_run_id,
                "target_generation": str(target_generation or ""),
                "max_attempts": max(1, int(max_attempts)),
            },
        ).scalar_one()
        self.session.expire_all()
        job = self.session.get(Job, int(job_id))
        if job is None:
            raise RuntimeError("coalescing job upsert did not return a persisted row")
        return job

    def claim_coalescing_job(
        self,
        *,
        job_type: str,
        worker_id: str,
        now: datetime,
        lease_seconds: int,
        target_generation: str | None = None,
        include_derived_generations: bool = False,
    ) -> Job | None:
        claimed_at = _normalize_utc_sqlite_timestamp(now)
        lease_until = _normalize_utc_sqlite_timestamp(
            now + timedelta(seconds=max(1, int(lease_seconds)))
        )
        generation_predicate = ""
        parameters: dict[str, Any] = {
            "job_type": job_type,
            "worker_id": worker_id,
            "claimed_at": claimed_at,
            "lease_until": lease_until,
        }
        if target_generation is not None:
            parameters["target_generation"] = str(target_generation)
            if include_derived_generations:
                parameters["derived_generation_pattern"] = (
                    f"{str(target_generation)}:late:%"
                )
                generation_predicate = (
                    "AND (target_generation = :target_generation "
                    "OR target_generation LIKE :derived_generation_pattern) "
                )
            else:
                generation_predicate = (
                    "AND target_generation = :target_generation "
                )
        job_id = self.session.execute(
            text(
                "UPDATE jobs SET "
                "status = 'running', locked_by = :worker_id, locked_at = :claimed_at, "
                "lease_until = :lease_until, claimed_generation = requested_generation "
                "WHERE id = ("
                "SELECT id FROM jobs "
                "WHERE job_type = :job_type AND status = 'queued' AND run_at <= :claimed_at "
                f"{generation_predicate}"
                "ORDER BY run_at ASC, id ASC LIMIT 1"
                ") AND status = 'queued' RETURNING id"
            ),
            parameters,
        ).scalar_one_or_none()
        if job_id is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(job_id))

    def complete_coalescing_job(
        self,
        *,
        job_id: int,
        worker_id: str,
        claimed_generation: int,
        now: datetime,
    ) -> Job | None:
        completed_at = _normalize_utc_sqlite_timestamp(now)
        completed_id = self.session.execute(
            text(
                "UPDATE jobs SET "
                "processed_generation = CASE "
                "WHEN processed_generation < :claimed_generation THEN :claimed_generation "
                "ELSE processed_generation END, "
                "status = CASE "
                "WHEN requested_generation = :claimed_generation THEN 'completed' "
                "ELSE 'queued' END, "
                "run_at = CASE "
                "WHEN requested_generation = :claimed_generation THEN run_at "
                "ELSE :completed_at END, "
                "completed_at = CASE "
                "WHEN requested_generation = :claimed_generation THEN :completed_at "
                "ELSE NULL END, "
                "locked_by = NULL, locked_at = NULL, lease_until = NULL "
                "WHERE id = :job_id AND status = 'running' "
                "AND locked_by = :worker_id AND claimed_generation = :claimed_generation "
                "RETURNING id"
            ),
            {
                "job_id": job_id,
                "worker_id": worker_id,
                "claimed_generation": claimed_generation,
                "completed_at": completed_at,
            },
        ).scalar_one_or_none()
        if completed_id is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(completed_id))

    def fail_coalescing_job(
        self,
        *,
        job_id: int,
        worker_id: str,
        claimed_generation: int,
        error_code: str,
        now: datetime,
        retry_at: datetime,
    ) -> Job | None:
        failed_id = self.session.execute(
            text(
                "UPDATE jobs SET "
                "attempt_count = attempt_count + 1, "
                "status = CASE WHEN attempt_count + 1 >= max_attempts "
                "THEN 'failed' ELSE 'queued' END, "
                "run_at = :retry_at, "
                "last_error_code = :error_code, "
                "completed_at = CASE WHEN attempt_count + 1 >= max_attempts "
                "THEN :failed_at ELSE NULL END, "
                "locked_by = NULL, locked_at = NULL, lease_until = NULL "
                "WHERE id = :job_id AND status = 'running' "
                "AND locked_by = :worker_id "
                "AND claimed_generation = :claimed_generation "
                "RETURNING id"
            ),
            {
                "job_id": int(job_id),
                "worker_id": str(worker_id),
                "claimed_generation": int(claimed_generation),
                "error_code": str(error_code or "")[:96],
                "retry_at": _normalize_utc_sqlite_timestamp(retry_at),
                "failed_at": _normalize_utc_sqlite_timestamp(now),
            },
        ).scalar_one_or_none()
        if failed_id is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(failed_id))

    def recover_failed_episode_jobs(
        self,
        *,
        now: datetime,
        compaction_generation: str,
        enabled_group_ids: frozenset[int] | None,
        limit: int,
        attempts_per_cycle: int = 3,
        dry_run: bool = False,
    ) -> int:
        """Rearm due failures without erasing cumulative attempts or error history."""
        if enabled_group_ids is not None and not enabled_group_ids:
            return 0
        params: dict[str, Any] = {
            "now": _normalize_utc_sqlite_timestamp(now),
            "generation": compaction_generation,
            "limit": max(1, int(limit)),
            "attempts_per_cycle": max(1, int(attempts_per_cycle)),
        }
        group_filter = ""
        if enabled_group_ids is not None:
            group_filter = "AND e.group_id IN :group_ids "
            params["group_ids"] = tuple(sorted(enabled_group_ids))
        selector = (
            "SELECT j.id FROM jobs j "
            "JOIN conversation_episodes e ON e.id=CAST(json_extract(j.payload_json, '$.episode_id') AS INTEGER) "
            "WHERE j.job_type='memory_episode_process' AND j.status='failed' "
            "AND datetime(j.run_at) <= datetime(:now) "
            "AND j.target_generation=:generation "
            "AND e.group_id=CAST(json_extract(j.payload_json, '$.group_id') AS INTEGER) "
            "AND e.compaction_version=j.target_generation "
            "AND e.is_current=1 AND e.status='failed' "
            f"{group_filter}"
            "ORDER BY j.run_at, j.id LIMIT :limit"
        )
        if dry_run:
            statement = text(selector)
        else:
            statement = text(
                "UPDATE jobs SET status='queued', "
                "max_attempts=attempt_count + :attempts_per_cycle, "
                "completed_at=NULL, locked_by=NULL, locked_at=NULL, lease_until=NULL "
                f"WHERE id IN ({selector}) "
                "AND status='failed' RETURNING id"
            )
        if enabled_group_ids is not None:
            statement = statement.bindparams(bindparam("group_ids", expanding=True))
        count = len(self.session.execute(statement, params).scalars().all())
        self.session.expire_all()
        return count

    def update_coalescing_job_payload(
        self,
        *,
        job_id: int,
        worker_id: str,
        claimed_generation: int,
        payload_json: dict[str, Any],
    ) -> Job | None:
        updated_id = self.session.execute(
            text(
                "UPDATE jobs SET payload_json = :payload_json "
                "WHERE id = :job_id AND status = 'running' "
                "AND locked_by = :worker_id "
                "AND claimed_generation = :claimed_generation "
                "RETURNING id"
            ),
            {
                "job_id": int(job_id),
                "worker_id": str(worker_id),
                "claimed_generation": int(claimed_generation),
                "payload_json": json.dumps(payload_json),
            },
        ).scalar_one_or_none()
        if updated_id is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(updated_id))

    def retry_failed_coalescing_job(
        self,
        *,
        job_id: int,
        run_at: datetime,
        reset_attempts: bool = True,
    ) -> Job | None:
        result = self.session.execute(
            text(
                "UPDATE jobs SET status = 'queued', run_at = :run_at, "
                "attempt_count = CASE WHEN :reset_attempts THEN 0 ELSE attempt_count END, "
                "last_error_code = '', completed_at = NULL, "
                "locked_by = NULL, locked_at = NULL, lease_until = NULL "
                "WHERE id = :job_id AND status = 'failed' RETURNING id"
            ),
            {
                "job_id": int(job_id),
                "run_at": _normalize_utc_sqlite_timestamp(run_at),
                "reset_attempts": bool(reset_attempts),
            },
        ).scalar_one_or_none()
        if result is None:
            return None
        self.session.expire_all()
        return self.session.get(Job, int(result))

    def requeue_stale_coalescing_jobs(
        self,
        *,
        job_type: str,
        now: datetime,
    ) -> int:
        stale_before = _normalize_utc_sqlite_timestamp(now)
        result = self.session.execute(
            text(
                "UPDATE jobs SET status = 'queued', run_at = :stale_before, "
                "locked_by = NULL, locked_at = NULL, lease_until = NULL "
                "WHERE job_type = :job_type AND status = 'running' "
                "AND lease_until IS NOT NULL AND lease_until <= :stale_before"
            ),
            {"job_type": job_type, "stale_before": stale_before},
        )
        self.session.expire_all()
        return int(result.rowcount or 0)

    def count_active_jobs(self, *, job_type: str, statuses: list[str] | None = None) -> int:
        active_statuses = statuses or ["queued", "running"]
        stmt = select(func.count(Job.id)).where(Job.job_type == job_type, Job.status.in_(active_statuses))
        return int(self.session.execute(stmt).scalar_one() or 0)

    def list_jobs(self, *, job_type: str, statuses: list[str]) -> list[Job]:
        if not statuses:
            return []
        stmt = (
            select(Job)
            .where(Job.job_type == job_type, Job.status.in_(statuses))
            .order_by(Job.id.asc())
        )
        return list(self.session.scalars(stmt))

    def cancel_jobs(
        self,
        *,
        job_type: str,
        statuses: Sequence[str],
        now: datetime,
    ) -> int:
        """Cancel a bounded job family and release any stale lease state."""
        normalized_statuses = tuple(
            dict.fromkeys(str(status) for status in statuses if str(status))
        )
        if not normalized_statuses:
            return 0
        rows = list(
            self.session.scalars(
                select(Job).where(
                    Job.job_type == job_type,
                    Job.status.in_(normalized_statuses),
                )
            )
        )
        completed_at = _normalize_utc_sqlite_timestamp(now)
        for row in rows:
            row.status = "cancelled"
            row.locked_by = None
            row.locked_at = None
            row.lease_until = None
            row.completed_at = completed_at
        self.session.flush()
        return len(rows)

    def claim_oldest_queued_job(self, *, job_type: str, now: datetime | None = None) -> Job | None:
        run_before = _normalize_utc_sqlite_timestamp(now or datetime.now().astimezone())
        lease_until = _normalize_utc_sqlite_timestamp(
            (now or datetime.now(UTC)) + timedelta(minutes=15)
        )
        job_id = self.session.execute(
            text(
                "UPDATE jobs SET status = 'running', run_at = :lease_until, "
                "locked_by = 'legacy-worker', locked_at = :run_before, lease_until = :lease_until "
                "WHERE id = ("
                "SELECT id FROM jobs WHERE job_type = :job_type AND status = 'queued' AND run_at <= :run_before "
                "ORDER BY id ASC LIMIT 1"
                ") AND status = 'queued' RETURNING id"
            ),
            {"job_type": job_type, "run_before": run_before, "lease_until": lease_until},
        ).scalar_one_or_none()
        if job_id is None:
            return None
        return self.session.get(Job, int(job_id))

    def mark_job_status(self, *, job_id: int, status: str, payload_json: dict[str, Any] | None = None) -> Job | None:
        job = self.session.get(Job, job_id)
        if job is None:
            return None
        job.status = status
        if payload_json is not None:
            job.payload_json = payload_json
        self.session.add(job)
        return job

    def retry_job(self, *, job_id: int, payload_json: dict[str, Any], run_at: datetime) -> Job | None:
        job = self.session.get(Job, job_id)
        if job is None:
            return None
        job.status = "queued"
        job.payload_json = payload_json
        job.run_at = run_at
        self.session.add(job)
        return job

    def next_queued_job_at(self, *, job_type: str) -> datetime | None:
        value = self.session.execute(
            select(func.min(Job.run_at)).where(Job.job_type == job_type, Job.status == "queued")
        ).scalar_one_or_none()
        return value

    def requeue_running_jobs(self, *, job_type: str) -> int:
        jobs = self.list_jobs(job_type=job_type, statuses=["running"])
        for job in jobs:
            job.status = "queued"
            self.session.add(job)
        return len(jobs)

    def requeue_stale_running_jobs(self, *, job_type: str, now: datetime | None = None) -> int:
        stale_before = _normalize_utc_sqlite_timestamp(now or datetime.now(UTC))
        jobs = list(
            self.session.scalars(
                select(Job).where(
                    Job.job_type == job_type,
                    Job.status == "running",
                    func.coalesce(Job.lease_until, Job.run_at) <= stale_before,
                )
            )
        )
        for job in jobs:
            job.status = "queued"
            job.locked_by = None
            job.locked_at = None
            job.lease_until = None
            self.session.add(job)
        return len(jobs)

class UsageRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_usage(
        self,
        *,
        timestamp: datetime,
        model: str,
        endpoint: str,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
    ) -> UsageRecord:
        record = UsageRecord(
            timestamp=_normalize_utc_sqlite_timestamp(timestamp),
            model=model,
            endpoint=endpoint,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )
        self.session.add(record)
        return record

    def summarize_usage(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        model: str | None = None,
    ) -> dict[str, int]:
        normalized_start_at = _normalize_utc_sqlite_timestamp(start_at)
        normalized_end_at = _normalize_utc_sqlite_timestamp(end_at)
        stmt = select(
            func.count(UsageRecord.id),
            func.sum(UsageRecord.input_tokens),
            func.sum(UsageRecord.cached_input_tokens),
            func.sum(UsageRecord.output_tokens),
        ).where(
            UsageRecord.timestamp >= normalized_start_at,
            UsageRecord.timestamp <= normalized_end_at,
        )
        if model is not None:
            stmt = stmt.where(UsageRecord.model == model)
        call_count, input_tokens, cached_input_tokens, output_tokens = self.session.execute(stmt).one()
        return {
            "call_count": int(call_count or 0),
            "input_tokens": int(input_tokens or 0),
            "cached_input_tokens": int(cached_input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        }
