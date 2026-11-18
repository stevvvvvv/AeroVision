from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, create_engine, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import get_settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    part_reference: Mapped[str | None] = mapped_column(String(120))
    original_filename: Mapped[str | None] = mapped_column(String(255))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)

    category: Mapped[str] = mapped_column(String(64))
    category_confidence: Mapped[float | None] = mapped_column(Float)
    category_source: Mapped[str] = mapped_column(String(32))     # classifier | user | single_model | none
    engine: Mapped[str] = mapped_column(String(32))              # patchcore | baseline_opencv
    model_version: Mapped[str] = mapped_column(String(120))
    calibration_source: Mapped[str] = mapped_column(String(32))

    anomaly_score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)              # P(défaut) calibrée
    verdict: Mapped[str] = mapped_column(String(32), index=True)  # conforme_probable | a_verifier | non_conforme_probable
    status: Mapped[str] = mapped_column(String(32), index=True, default="pending_review")
    final_decision: Mapped[str | None] = mapped_column(String(32))  # après revue humaine
    processing_ms: Mapped[int] = mapped_column(Integer)
    warnings: Mapped[list] = mapped_column(JSON, default=list)

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan", order_by="Detection.rank")
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="inspection", cascade="all, delete-orphan", order_by="Review.created_at")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id", ondelete="CASCADE"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    w: Mapped[int] = mapped_column(Integer)
    h: Mapped[int] = mapped_column(Integer)
    area_px: Mapped[int] = mapped_column(Integer)
    max_score: Mapped[float] = mapped_column(Float)
    mean_score: Mapped[float] = mapped_column(Float)
    polygon: Mapped[list] = mapped_column(JSON, default=list)
    defect_type: Mapped[str] = mapped_column(String(32), default="unknown")
    defect_type_confidence: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32))

    inspection: Mapped[Inspection] = relationship(back_populates="detections")


class Review(Base):
    """Revue humaine — en ajout seul (append-only) pour garder la traçabilité complète."""
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    inspection_id: Mapped[str] = mapped_column(ForeignKey("inspections.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reviewer: Mapped[str] = mapped_column(String(120))
    decision: Mapped[str] = mapped_column(String(32))   # conforme | non_conforme | inconclusif
    comment: Mapped[str | None] = mapped_column(Text)
    detection_feedback: Mapped[list] = mapped_column(JSON, default=list)
    added_regions: Mapped[list] = mapped_column(JSON, default=list)
    model_verdict_at_review: Mapped[str] = mapped_column(String(32))

    inspection: Mapped[Inspection] = relationship(back_populates="reviews")


_settings = get_settings()
_connect_args = {"check_same_thread": False} if _settings.database_url.startswith("sqlite") else {}
engine = create_engine(_settings.database_url, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    if _settings.database_url.startswith("sqlite:///"):
        from pathlib import Path
        Path(_settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session
