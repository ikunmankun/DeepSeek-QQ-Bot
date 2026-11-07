from __future__ import annotations

from datetime import datetime

from app.core.time_utils import ASIA_SHANGHAI

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, column_property, mapped_column


class Base(DeclarativeBase):
    pass


def _utc_now() -> datetime:
    return datetime.now(ASIA_SHANGHAI).replace(tzinfo=None)


class Group(Base):
    __tablename__ = "groups"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_name: Mapped[str] = mapped_column(String(255), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    speak_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    reply_mode: Mapped[str] = mapped_column(String(32), default="balanced")
    cooldown_seconds: Mapped[int] = mapped_column(Integer, default=90)
    persona_variant: Mapped[str] = mapped_column(String(64), default="default")


class GroupPersonaState(Base):
    """Per-group active persona plus one account-scoped row for profile snapshots.

    ``group_id=0`` is reserved for account-level state (the original QQ
    avatar). Regular rows store a group's active persona key and the original
    group card captured before the first persona switch.
    """

    __tablename__ = "group_persona_states"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    persona_key: Mapped[str] = mapped_column(String(64), default="default")
    card_snapshot: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class PersonaStyleExample(Base):
    """A verbatim member message plus its conversation context."""

    __tablename__ = "persona_style_examples"
    __table_args__ = (
        Index("ix_persona_style_examples_user_time", "user_id", "timestamp"),
    )

    group_id: Mapped[int] = mapped_column(Integer)
    user_id: Mapped[int] = mapped_column(Integer)
    msg_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    text: Mapped[str] = mapped_column(Text, default="")
    context_before: Mapped[list] = mapped_column(JSON, default=list)
    context_after: Mapped[list] = mapped_column(JSON, default=list)
    reply_target: Mapped[str | None] = mapped_column(String(512), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class PersonaStyleSyncState(Base):
    """Watermark and refresh bookkeeping for one member's live style sync."""

    __tablename__ = "persona_style_sync_state"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_msg_id: Mapped[str] = mapped_column(String(128), default="")
    last_refresh_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    new_since_refresh: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class PersonaExampleVector(Base):
    """Persistent semantic vectors for persona style examples.

    Kept separate from the memory-system vector tables so the persona
    retrieval cache survives restarts and never collides with memory
    retrieval (retrieval_documents_vec_* / memory_items_vec).
    """

    __tablename__ = "persona_example_vectors"
    __table_args__ = (
        Index("ix_pev_user", "user_id"),
    )

    msg_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer)
    group_id: Mapped[int] = mapped_column(Integer)
    provider: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(128), default="")
    embedding_version: Mapped[str] = mapped_column(String(64), default="")
    dimensions: Mapped[int] = mapped_column(Integer, default=0)
    document_schema: Mapped[str] = mapped_column(String(64), default="")
    document_hash: Mapped[str] = mapped_column(String(64), default="")
    vector_json: Mapped[str] = mapped_column(Text, default="[]")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utc_now,
    )


class MemberFactRefreshState(Base):
    """Independent watermark for member fact refresh.

    Kept separate from persona_style_sync_state so fact refresh can never
    advance (and thereby suppress) the style-sample sync watermark.
    """

    __tablename__ = "member_fact_refresh_state"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    last_msg_id: Mapped[str] = mapped_column(String(128), default="")
    last_refresh_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        Index("ix_users_nickname", "nickname"),
        Index("ix_users_group_card", "group_card"),
    )

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    nickname: Mapped[str] = mapped_column(String(255), default="")
    group_card: Mapped[str] = mapped_column(String(255), default="")
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    profile_summary: Mapped[str] = mapped_column(Text, default="")


class ContextReset(Base):
    """An authored context reset, separate from the immutable chat ledger."""
    __tablename__ = 'context_resets'
    source_id: Mapped[int] = mapped_column(ForeignKey('messages.id'), primary_key=True)
    account_id: Mapped[int] = mapped_column(Integer)
    scope: Mapped[str] = mapped_column(String(160), index=True)
    requested_count: Mapped[int] = mapped_column(Integer)
    excluded_count: Mapped[int] = mapped_column(Integer)
    mode: Mapped[str] = mapped_column(String(20), default='legacy_count', server_default='legacy_count')
    automatic_after_id: Mapped[int] = mapped_column(Integer, default=0, server_default='0')
    target_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class ContextExcludedMessage(Base):
    __tablename__ = 'context_excluded_messages'
    message_id: Mapped[int] = mapped_column(ForeignKey('messages.id'), primary_key=True)
    reset_source_id: Mapped[int] = mapped_column(ForeignKey('context_resets.source_id'), index=True)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("id", "group_id", name="ux_messages_id_group"),
        Index("ix_messages_group_reply", "group_id", "reply_to_msg_id"),
        Index("ix_messages_group_user", "group_id", "user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform_msg_id: Mapped[str] = mapped_column(String(128), unique=True)
    group_id: Mapped[int | None] = mapped_column(ForeignKey("groups.group_id"), nullable=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_json: Mapped[dict] = mapped_column(JSON)
    plain_text: Mapped[str] = mapped_column(Text, default="")
    msg_type: Mapped[str] = mapped_column(String(32), default="text")
    reply_to_msg_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    mentioned_bot: Mapped[bool] = mapped_column(Boolean, default=False)
    # A rebuildable context mask; original text, JSON and delivery state stay intact.
    context_excluded = column_property(select(ContextExcludedMessage.message_id).where(
        ContextExcludedMessage.message_id == id).correlate_except(ContextExcludedMessage).exists())


class Summary(Base):
    __tablename__ = "summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(String(32))
    scope_id: Mapped[str] = mapped_column(String(64))
    summary_level: Mapped[str] = mapped_column(String(32))
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text)
    source_count: Mapped[int] = mapped_column(Integer)
    # A stable topic or range key lets recursive summarization update its output
    # without retaining duplicate summaries for the same source material.
    summary_key: Mapped[str] = mapped_column(String(255), default="", index=True)
    source_start_msg_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_end_msg_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_summary_ids: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(32), default="active")


class MemoryItem(Base):
    __tablename__ = "memory_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(String(32))
    scope_id: Mapped[str] = mapped_column(String(64))
    subject_type: Mapped[str] = mapped_column(String(32))
    subject_id: Mapped[str] = mapped_column(String(64))
    memory_kind: Mapped[str] = mapped_column(String(32))
    canonical_key: Mapped[str] = mapped_column(String(255), default="", index=True)
    predicate: Mapped[str] = mapped_column(String(96), default="")
    object_text: Mapped[str] = mapped_column(Text, default="")
    content: Mapped[str] = mapped_column(Text)
    importance: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    source_msg_id: Mapped[str] = mapped_column(String(128))
    source_msg_ids: Mapped[list] = mapped_column(JSON, default=list)
    mention_count: Mapped[int] = mapped_column(Integer, default=1)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Keep extraction provenance and the validity lifecycle separate: an old
    # fact remains auditable after a newer fact supersedes it.
    valid_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    supersedes_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    superseded_by_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class MemoryItemSemanticVector(Base):
    """Persistent precomputed semantic vectors for memory_items."""

    __tablename__ = "memory_item_semantic_vectors"
    __table_args__ = (
        Index("ix_misv_group_memory", "group_id", "memory_id"),
    )

    memory_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(Integer)
    provider: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(128), default="")
    dimensions: Mapped[int] = mapped_column(Integer, default=0)
    version: Mapped[str] = mapped_column(String(64), default="")
    vector_json: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(ASIA_SHANGHAI).replace(tzinfo=None),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String(64))
    job_key: Mapped[str] = mapped_column(String(255), default="", index=True)
    payload_json: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    run_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    requested_generation: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    processed_generation: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    claimed_generation: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    attempt_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer, default=3, server_default="3")
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_code: Mapped[str] = mapped_column(String(96), default="", server_default="")
    backfill_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_generation: Mapped[str] = mapped_column(String(128), default="", server_default="")


class GroupMoodState(Base):
    """每群一份的运行时情绪状态（规则版）。

    事件触发（被夸/被怼/求助）抬升 intensity 并切换 mood，读取时按
    half-life 惰性衰减回 ambient（夜间 sleepy / 平时 default）。每群一行。
    """

    __tablename__ = "group_mood_states"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    mood: Mapped[str] = mapped_column(String(32), default="default")
    intensity: Mapped[float] = mapped_column(Float, default=0.0)
    reason: Mapped[str] = mapped_column(String(160), default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(ASIA_SHANGHAI).replace(tzinfo=None),
    )


class UserAffinityState(Base):
    """每群每人的好感度（规则版，麦麦式关系系统的轻量实现）。

    被夸/求助/日常互动加分，被怼小扣分；读取时按半衰期（约 14 天）向 0
    衰减。分数只用来定性（陌生/认识/熟人/老朋友），不搞数值化的亲密表演。
    """

    __tablename__ = "user_affinity_states"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    interactions: Mapped[int] = mapped_column(Integer, default=0)
    last_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(ASIA_SHANGHAI).replace(tzinfo=None),
    )


class GroupSlangTerm(Base):
    """群黑话/高频词（slang learning 落表，prompt 注入用）。"""

    __tablename__ = "group_slang_terms"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    term: Mapped[str] = mapped_column(String(32), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    last_seen: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(ASIA_SHANGHAI).replace(tzinfo=None),
    )


class AgentAgendaItem(Base):
    """agent 自己的日程/待办（夜间反思生成，观察轮推进）。"""

    __tablename__ = "agent_agenda_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16), default="todo")      # todo | curiosity
    content: Mapped[str] = mapped_column(String(300))
    source: Mapped[str] = mapped_column(String(16), default="self")    # self | owner
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)  # pending | done | dropped
    group_id: Mapped[int] = mapped_column(Integer, default=0)          # 0 = 通用
    note: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(ASIA_SHANGHAI).replace(tzinfo=None),
    )
    done_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AgentProject(Base):
    """agent 的长时程自主项目：跨天推进、有交付物。"""

    __tablename__ = "agent_projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(120))
    plan: Mapped[str] = mapped_column(String(600), default="")
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)  # active | done | shelved
    deliverable: Mapped[str] = mapped_column(String(400), default="")
    work_log: Mapped[list] = mapped_column(JSON, default=list)  # [{date, note}]
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(ASIA_SHANGHAI).replace(tzinfo=None),
    )
    last_worked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ConversationEpisode(Base):
    __tablename__ = "conversation_episodes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('open','closed','processing','processed','ready','failed','superseded')",
            name="ck_conversation_episodes_status",
        ),
        ForeignKeyConstraint(
            ["start_message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_conversation_episodes_start_message_group",
        ),
        ForeignKeyConstraint(
            ["end_message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_conversation_episodes_end_message_group",
        ),
        UniqueConstraint("id", "group_id", name="ux_conversation_episodes_id_group"),
        UniqueConstraint(
            "group_id",
            "segmentation_version",
            "start_message_id",
            "end_message_id",
            name="ux_conversation_episodes_identity",
        ),
        Index(
            "ux_conversation_episodes_open_group",
            "group_id",
            unique=True,
            sqlite_where=text("status = 'open' AND is_current = 1"),
        ),
        Index(
            "ix_conversation_episodes_group_time",
            "group_id",
            "started_at",
            "ended_at",
            "id",
        ),
        Index(
            "ix_conversation_episodes_group_status",
            "group_id",
            "status",
            "ended_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.group_id"), nullable=False)
    segmentation_version: Mapped[str] = mapped_column(String(64), default="v2")
    status: Mapped[str] = mapped_column(String(32), default="open")
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    start_message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    end_message_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    boundary_reason: Mapped[str] = mapped_column(String(64), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(128), default="")
    compaction_version: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EpisodeMessage(Base):
    __tablename__ = "episode_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["episode_id", "group_id"],
            ["conversation_episodes.id", "conversation_episodes.group_id"],
            name="fk_episode_messages_episode_group",
        ),
        ForeignKeyConstraint(
            ["message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_episode_messages_message_group",
        ),
        UniqueConstraint("episode_id", "message_id", name="ux_episode_messages_membership"),
        UniqueConstraint("episode_id", "ordinal", name="ux_episode_messages_ordinal"),
        UniqueConstraint("message_id", name="ux_episode_messages_message"),
        Index("ix_episode_messages_group_message", "group_id", "message_id"),
    )

    episode_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    group_id: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class RetrievalDocument(Base):
    __tablename__ = "retrieval_documents"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','active','inactive','superseded','failed')",
            name="ck_retrieval_documents_status",
        ),
        CheckConstraint(
            "embedding_status IN ('pending','ready','failed','disabled','stale')",
            name="ck_retrieval_documents_embedding_status",
        ),
        ForeignKeyConstraint(
            ["episode_id", "group_id"],
            ["conversation_episodes.id", "conversation_episodes.group_id"],
            name="fk_retrieval_documents_episode_group",
        ),
        UniqueConstraint("id", "group_id", name="ux_retrieval_documents_id_group"),
        UniqueConstraint(
            "scope_type",
            "scope_id",
            "document_kind",
            "source_table",
            "source_id",
            "content_hash",
            name="ux_retrieval_documents_source_version",
        ),
        Index(
            "ix_retrieval_documents_group_status_time",
            "group_id",
            "status",
            "start_at",
            "end_at",
        ),
        Index(
            "ix_retrieval_documents_episode",
            "group_id",
            "episode_id",
            "document_kind",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope_type: Mapped[str] = mapped_column(String(32))
    scope_id: Mapped[str] = mapped_column(String(64))
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.group_id"), nullable=False)
    episode_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    document_kind: Mapped[str] = mapped_column(String(64))
    source_table: Mapped[str] = mapped_column(String(64))
    source_id: Mapped[str] = mapped_column(String(255))
    start_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(32), default="active")
    embedding_provider: Mapped[str] = mapped_column(String(64), default="")
    embedding_model: Mapped[str] = mapped_column(String(255), default="")
    embedding_version: Mapped[str] = mapped_column(String(128), default="")
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Only window/message evidence that is part of the vector contract is
    # eligible. Summary/fact documents are intentionally FTS-only unless a
    # future embedding policy opts them in explicitly.
    embedding_eligible: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    embedding_status: Mapped[str] = mapped_column(String(32), default="disabled")
    last_error_code: Mapped[str] = mapped_column(String(96), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class RetrievalDocumentMessage(Base):
    __tablename__ = "retrieval_document_messages"
    __table_args__ = (
        ForeignKeyConstraint(
            ["document_id", "group_id"],
            ["retrieval_documents.id", "retrieval_documents.group_id"],
            name="fk_retrieval_document_messages_document_group",
        ),
        ForeignKeyConstraint(
            ["message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_retrieval_document_messages_message_group",
        ),
        UniqueConstraint(
            "document_id",
            "message_id",
            "role",
            name="ux_retrieval_document_messages_source",
        ),
        UniqueConstraint(
            "document_id",
            "ordinal",
            "role",
            name="ux_retrieval_document_messages_ordinal",
        ),
        Index("ix_retrieval_document_messages_group_message", "group_id", "message_id"),
        Index(
            "ix_retrieval_document_messages_group_document",
            "group_id",
            "document_id",
        ),
    )

    document_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(32), primary_key=True, default="source")
    group_id: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)


class RetrievalIndexState(Base):
    __tablename__ = "retrieval_index_state"
    __table_args__ = (
        UniqueConstraint("channel", "generation", name="ux_retrieval_index_state_generation"),
        Index(
            "ux_retrieval_index_state_active_channel",
            "channel",
            "document_family",
            unique=True,
            sqlite_where=text("is_active = 1"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    channel: Mapped[str] = mapped_column(String(32))
    generation: Mapped[int] = mapped_column(Integer)
    physical_table: Mapped[str] = mapped_column(String(255))
    provider: Mapped[str] = mapped_column(String(64), default="")
    model: Mapped[str] = mapped_column(String(255), default="")
    dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    version: Mapped[str] = mapped_column(String(128), default="")
    status: Mapped[str] = mapped_column(String(32), default="building")
    total_documents: Mapped[int] = mapped_column(Integer, default=0)
    indexed_documents: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    document_family: Mapped[str] = mapped_column(String(64), default="")
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class MemoryBackfillRun(Base):
    __tablename__ = "memory_backfill_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','completed','failed','cancelled')",
            name="ck_memory_backfill_runs_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_key: Mapped[str] = mapped_column(String(255), unique=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    snapshot_watermarks_json: Mapped[dict] = mapped_column(JSON, default=dict)
    manifest_json: Mapped[dict] = mapped_column(JSON, default=dict)
    segmentation_generation: Mapped[str] = mapped_column(String(128), default="")
    compaction_generation: Mapped[str] = mapped_column(String(128), default="")
    index_generation: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_code: Mapped[str] = mapped_column(String(96), default="")


class MemoryLateArrivalPreparation(Base):
    """Idempotency ledger for destructive late-arrival resegmentation."""

    __tablename__ = "memory_late_arrival_preparations"
    __table_args__ = (
        ForeignKeyConstraint(
            ["message_id", "group_id"],
            ["messages.id", "messages.group_id"],
            name="fk_memory_late_arrival_message_group",
        ),
        UniqueConstraint(
            "group_id",
            "message_id",
            "segmentation_generation",
            name="ux_memory_late_arrival_preparation_key",
        ),
        Index(
            "ix_memory_late_arrival_preparations_group_message",
            "group_id",
            "message_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.group_id"), nullable=False)
    message_id: Mapped[int] = mapped_column(Integer, nullable=False)
    segmentation_generation: Mapped[str] = mapped_column(String(128))
    compaction_generation: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class UsageRecord(Base):
    __tablename__ = "usage_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model: Mapped[str] = mapped_column(String(64))
    endpoint: Mapped[str] = mapped_column(String(32))
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)


class BbotListenerCacheEntry(Base):
    __tablename__ = "bbot_listener_cache_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(Integer, index=True)
    platform: Mapped[str] = mapped_column(String(32), index=True)
    external_id: Mapped[str] = mapped_column(String(128), index=True)
    canonical_name: Mapped[str] = mapped_column(String(255), default="")
    aliases_json: Mapped[list] = mapped_column(JSON, default=list)
    source: Mapped[str] = mapped_column(String(64), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DevSession(Base):
    """Private-chat turn store.

    The table predates the chat-only private service (2026-09-11): it used to
    hold Codex project sessions, and now backs daily private conversations and
    private drawings.  The schema is kept without migration so existing rows
    stay readable; ``session_mode`` is ``daily`` for every new session.
    """

    __tablename__ = "dev_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_qq: Mapped[int] = mapped_column(Integer, index=True)
    session_mode: Mapped[str] = mapped_column(String(32), default="project")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_active_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    summary: Mapped[str] = mapped_column(Text, default="")
    last_task_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class DevTask(Base):
    """One private-chat turn (or private image job) inside a dev session.

    ``intent_type`` is ``private_chat`` for chat turns; the file/command and
    restart columns are legacy and stay empty.
    """

    __tablename__ = "dev_tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("dev_sessions.id"), index=True)
    requested_by_qq: Mapped[int] = mapped_column(Integer, index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_request_text: Mapped[str] = mapped_column(Text)
    intent_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    summary: Mapped[str] = mapped_column(Text, default="")
    files_read_json: Mapped[list] = mapped_column(JSON, default=list)
    files_changed_json: Mapped[list] = mapped_column(JSON, default=list)
    commands_run_json: Mapped[list] = mapped_column(JSON, default=list)
    restart_required: Mapped[bool] = mapped_column(Boolean, default=False)
    restart_result: Mapped[str] = mapped_column(String(64), default="")
    failure_reason: Mapped[str] = mapped_column(Text, default="")
    checkpoint_dir: Mapped[str] = mapped_column(Text, default="")
    result_text: Mapped[str] = mapped_column(Text, default="")


class DevTaskArtifact(Base):
    __tablename__ = "dev_task_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("dev_tasks.id"), index=True)
    artifact_type: Mapped[str] = mapped_column(String(64))
    artifact_path: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class PersonaTraitCandidate(Base):
    """特质证据表：人格特质经"多次跨天证据"确认后才能进入提示词。

    证据制人格进化（对照 dsh-identity 的思路）：特质不是设定出来的，是从
    行为记录里长出来的——一条候选特质必须被证据命中至少 PROMOTION 次、
    且证据跨至少 PROMOTION_SPAN_DAYS 天，才从 candidate 晋升为 confirmed。
    证据来源是机器人自己的日记与关于其行为的记忆，由周期 worker 用 LLM
    归纳；这里只存结论与计数。
    """

    __tablename__ = "persona_trait_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # 归一化指纹（去空白小写后的摘要），用来把同一特质的不同措辞合并计数。
    trait_key: Mapped[str] = mapped_column(String(64), unique=True)
    display_text: Mapped[str] = mapped_column(String(120))
    origin: Mapped[str] = mapped_column(String(24), default="diary")
    evidence_count: Mapped[int] = mapped_column(Integer, default=1)
    evidence_json: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="candidate", index=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class GroupInterjectionOutcome(Base):
    """主动插话的结果记录（沉默学习）。

    每次机器人主动插话（judge 批准的 proactive 回复）落一行 pending，
    周期 worker 在观察窗口过后回填结果：有人接话 responded、被怼 negative、
    没人理 ignored、没发出去 not_sent。按群 × 时段聚合出"什么时候插话
    没人接"，长期喂给插话判定当参考。
    """

    __tablename__ = "group_interjection_outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int] = mapped_column(Integer, index=True)
    bot_msg_id: Mapped[str] = mapped_column(String(128), unique=True)
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    hour_bucket: Mapped[int] = mapped_column(Integer)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class BilibiliGroupInterest(Base):
    """群的 B 站兴趣画像：按视频分区聚合，随时间衰减。

    群友分享过什么视频，机器人就慢慢知道这个群在看什么——
    分区（tname）计分，读取时按半衰期向 0 衰减，久没人聊的口味自然淡出。
    """

    __tablename__ = "bilibili_group_interests"

    group_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    zone: Mapped[str] = mapped_column(String(48), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    share_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Reminder(Base):
    """提醒/闹钟：到点在群里 @人（或私聊），没人应就贪睡式再催。

    一次性（once）催满 max_nags 次没人理就放弃；每天（daily）的催完一轮
    自动顺延到明天同一时刻继续。channel=group 走群 @，private 走私聊。
    """

    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    group_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    channel: Mapped[str] = mapped_column(String(16), default="group")
    content: Mapped[str] = mapped_column(String(200), default="")
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    repeat: Mapped[str] = mapped_column(String(16), default="once")
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    nag_count: Mapped[int] = mapped_column(Integer, default=0)
    max_nags: Mapped[int] = mapped_column(Integer, default=2)
    next_nag_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class LifeNote(Base):
    """人生笔记：bot 的自传体长期记忆。

    事实层记忆（memory_items）会衰减、会过期，日记是散落的文件——
    这里是夜间反思把"这段时间发生了什么、我们的交情走到哪了"凝练成的
    叙事条目：只增不灭、全局共享（群+私聊是同一个人格）。供提示注入
    与 FTS/向量检索，是"越聊越像老朋友"的那一层。
    """

    __tablename__ = "life_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    period: Mapped[str] = mapped_column(String(64), default="")          # 如 2026-W45
    kind: Mapped[str] = mapped_column(String(16), default="milestone")   # relationship|running_joke|group_history|self_growth|promise|milestone
    title: Mapped[str] = mapped_column(String(120), default="")
    content: Mapped[str] = mapped_column(Text, default="")
    sources: Mapped[list] = mapped_column(JSON, default=list)            # 日记日期 / memory_item id 等出处
    source_groups: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)  # active|superseded
    embedding_provider: Mapped[str] = mapped_column(String(64), default="")
    embedding_model: Mapped[str] = mapped_column(String(64), default="")
    embedding_dimensions: Mapped[int] = mapped_column(Integer, default=0)
    embedding_version: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
