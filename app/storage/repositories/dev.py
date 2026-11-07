"""dev 域的仓储类（Phase 1a 自 repositories.py 拆出，内容未改动）。"""

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

class DevSessionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_latest_owner_session(self, *, owner_qq: int, session_mode: str = "project") -> DevSession | None:
        # A late completion in an older daily session must not reactivate it
        # after a user explicitly opened a fresh context.
        ordering = ((DevSession.id.desc(),) if session_mode == 'daily'
            else (DevSession.last_active_at.desc(), DevSession.id.desc()))
        stmt = (
            select(DevSession)
            .where(DevSession.owner_qq == owner_qq, DevSession.session_mode == session_mode)
            .order_by(*ordering)
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create_owner_session(self, *, owner_qq: int, session_mode: str = "project") -> DevSession:
        now = shanghai_now_naive()
        dev_session = DevSession(
            owner_qq=owner_qq,
            session_mode=session_mode,
            started_at=now,
            last_active_at=now,
            summary="",
        )
        self.session.add(dev_session)
        self.session.flush()
        return dev_session

    def get_or_create_owner_session(self, *, owner_qq: int, session_mode: str = "project") -> DevSession:
        dev_session = self.get_latest_owner_session(owner_qq=owner_qq, session_mode=session_mode)
        if dev_session is None:
            dev_session = self.create_owner_session(owner_qq=owner_qq, session_mode=session_mode)
        else:
            dev_session.last_active_at = shanghai_now_naive()
            self.session.add(dev_session)
        self.session.add(dev_session)
        self.session.flush()
        return dev_session

    def list_recent_owner_sessions(
        self,
        *,
        owner_qq: int,
        limit: int,
        session_modes: list[str] | None = None,
    ) -> list[DevSession]:
        stmt = select(DevSession).where(DevSession.owner_qq == owner_qq)
        if session_modes:
            stmt = stmt.where(DevSession.session_mode.in_(session_modes))
        stmt = stmt.order_by(DevSession.last_active_at.desc(), DevSession.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def update_session(
        self,
        *,
        session_id: int,
        summary: str | None = None,
        last_task_id: int | None = None,
    ) -> DevSession | None:
        dev_session = self.session.get(DevSession, session_id)
        if dev_session is None:
            return None
        dev_session.last_active_at = shanghai_now_naive()
        if summary is not None:
            dev_session.summary = summary
        if last_task_id is not None:
            dev_session.last_task_id = last_task_id
        self.session.add(dev_session)
        return dev_session

class DevTaskRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_task(
        self,
        *,
        session_id: int,
        requested_by_qq: int,
        raw_request_text: str,
        intent_type: str,
        status: str = "queued",
    ) -> DevTask:
        task = DevTask(
            session_id=session_id,
            requested_by_qq=requested_by_qq,
            requested_at=shanghai_now_naive(),
            raw_request_text=raw_request_text,
            intent_type=intent_type,
            status=status,
            summary="",
            files_read_json=[],
            files_changed_json=[],
            commands_run_json=[],
            restart_required=False,
            restart_result="",
            failure_reason="",
            checkpoint_dir="",
            result_text="",
        )
        self.session.add(task)
        self.session.flush()
        return task

    def list_tasks_by_status(self, status: str) -> list[DevTask]:
        stmt = select(DevTask).where(DevTask.status == status).order_by(DevTask.id.asc())
        return list(self.session.scalars(stmt))

    def list_tasks_for_session_by_status(self, *, session_id: int, statuses: list[str]) -> list[DevTask]:
        if not statuses:
            return []
        stmt = (
            select(DevTask)
            .where(DevTask.session_id == session_id, DevTask.status.in_(statuses))
            .order_by(DevTask.id.asc())
        )
        return list(self.session.scalars(stmt))

    def list_tasks_by_statuses(self, *, statuses: list[str], intent_types: list[str] | None = None) -> list[DevTask]:
        if not statuses:
            return []
        stmt = select(DevTask).where(DevTask.status.in_(statuses))
        if intent_types:
            stmt = stmt.where(DevTask.intent_type.in_(intent_types))
        stmt = stmt.order_by(DevTask.id.asc())
        return list(self.session.scalars(stmt))

    def list_recent_tasks_for_session(self, *, session_id: int, limit: int) -> list[DevTask]:
        stmt = (
            select(DevTask)
            .where(DevTask.session_id == session_id)
            .order_by(DevTask.id.desc())
            .limit(limit)
        )
        return list(reversed(list(self.session.scalars(stmt))))

    def claim_oldest_queued_task(self, *, intent_types: list[str] | None = None) -> DevTask | None:
        stmt = select(DevTask).where(DevTask.status == "queued")
        if intent_types:
            stmt = stmt.where(DevTask.intent_type.in_(intent_types))
        stmt = stmt.order_by(DevTask.id.asc()).limit(1)
        task = self.session.execute(stmt).scalar_one_or_none()
        if task is None:
            return None
        task.status = "running"
        self.session.add(task)
        self.session.flush()
        return task

    def get_task(self, task_id: int) -> DevTask | None:
        return self.session.get(DevTask, task_id)

    def mark_completed(
        self,
        *,
        task_id: int,
        summary: str,
        result_text: str,
        files_read: list[str],
        files_changed: list[str],
        commands_run: list[str],
        restart_required: bool,
        restart_result: str,
        checkpoint_dir: str,
    ) -> DevTask | None:
        task = self.session.get(DevTask, task_id)
        if task is None:
            return None
        task.status = "completed"
        task.summary = summary
        task.result_text = result_text
        task.files_read_json = files_read
        task.files_changed_json = files_changed
        task.commands_run_json = commands_run
        task.restart_required = restart_required
        task.restart_result = restart_result
        task.checkpoint_dir = checkpoint_dir
        self.session.add(task)
        return task

    def mark_failed(self, *, task_id: int, failure_reason: str, checkpoint_dir: str = "") -> DevTask | None:
        task = self.session.get(DevTask, task_id)
        if task is None:
            return None
        task.status = "failed"
        task.failure_reason = failure_reason
        if checkpoint_dir:
            task.checkpoint_dir = checkpoint_dir
        self.session.add(task)
        return task

    def mark_status(self, *, task_id: int, status: str) -> DevTask | None:
        task = self.session.get(DevTask, task_id)
        if task is None:
            return None
        task.status = status
        self.session.add(task)
        return task

class DevTaskArtifactRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def add_artifact(
        self,
        *,
        task_id: int,
        artifact_type: str,
        artifact_path: str,
        metadata_json: dict[str, Any],
    ) -> DevTaskArtifact:
        artifact = DevTaskArtifact(
            task_id=task_id,
            artifact_type=artifact_type,
            artifact_path=artifact_path,
            metadata_json=metadata_json,
        )
        self.session.add(artifact)
        self.session.flush()
        return artifact
