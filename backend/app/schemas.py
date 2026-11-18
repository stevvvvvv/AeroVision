from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["conforme", "non_conforme", "inconclusif"]


class DetectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    rank: int
    x: int
    y: int
    w: int
    h: int
    area_px: int
    max_score: float
    mean_score: float
    polygon: list
    defect_type: str
    defect_type_confidence: float | None
    source: str


class ReviewOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    reviewer: str
    decision: str
    comment: str | None
    detection_feedback: list
    added_regions: list
    model_verdict_at_review: str


class InspectionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    created_at: datetime
    part_reference: str | None
    original_filename: str | None
    category: str
    engine: str
    anomaly_score: float
    confidence: float
    verdict: str
    status: str
    final_decision: str | None
    n_detections: int = 0


class InspectionOut(InspectionSummary):
    width: int
    height: int
    category_confidence: float | None
    category_source: str
    model_version: str
    calibration_source: str
    processing_ms: int
    warnings: list[str]
    detections: list[DetectionOut]
    reviews: list[ReviewOut]
    aero_hints: dict[str, str] = {}


class DetectionFeedback(BaseModel):
    detection_id: int
    is_true_defect: bool
    corrected_type: str | None = None


class AddedRegion(BaseModel):
    """Défaut manqué par le modèle, signalé par l'opérateur (faux négatif)."""
    x: int
    y: int
    w: int
    h: int
    defect_type: str = "other"


class ReviewIn(BaseModel):
    reviewer: str = Field(min_length=2, max_length=120)
    decision: Decision
    comment: str | None = Field(default=None, max_length=4000)
    detection_feedback: list[DetectionFeedback] = []
    added_regions: list[AddedRegion] = []


class Page(BaseModel):
    total: int
    items: list[InspectionSummary]
