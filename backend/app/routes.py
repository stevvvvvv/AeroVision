from __future__ import annotations

import csv
import io
import uuid

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from aerovision_ml.taxonomy import GENERIC_DEFECTS, GENERIC_LABELS_FR, GENERIC_TO_AERO_HINT

from . import storage
from .config import get_settings
from .db import Detection, Inspection, Review, get_session
from .schemas import InspectionOut, InspectionSummary, Page, ReviewIn

router = APIRouter(prefix="/api")

VERDICT_PRIORITY = {"non_conforme_probable": 0, "a_verifier": 1, "conforme_probable": 2}


def _engine(request: Request):
    return request.app.state.engine


def _summary(i: Inspection) -> InspectionSummary:
    s = InspectionSummary.model_validate(i)
    s.n_detections = len(i.detections)
    return s


def _full(i: Inspection) -> InspectionOut:
    out = InspectionOut.model_validate(i)
    out.n_detections = len(i.detections)
    types = {d.defect_type for d in i.detections}
    out.aero_hints = {t: GENERIC_TO_AERO_HINT.get(t, "") for t in types}
    return out


def _get(session: Session, inspection_id: str) -> Inspection:
    i = session.get(Inspection, inspection_id, options=[selectinload(Inspection.detections),
                                                        selectinload(Inspection.reviews)])
    if i is None:
        raise HTTPException(404, "Inspection introuvable")
    return i


# ---------------------------------------------------------------------- système
@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/system")
def system(request: Request):
    return {**_engine(request).info(),
            "defect_labels_fr": GENERIC_LABELS_FR,
            "disclaimer": "Démonstrateur d'aide à l'inspection. Non qualifié pour la "
                          "libération de pièces aéronautiques ; décision finale humaine obligatoire."}


# ---------------------------------------------------------------------- inspections
@router.post("/inspections", response_model=InspectionOut, status_code=201)
async def create_inspection(
    request: Request,
    file: UploadFile = File(...),
    part_reference: str | None = Form(None),
    category: str | None = Form(None),
    session: Session = Depends(get_session),
):
    settings = get_settings()
    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image > {settings.max_upload_mb} Mo")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(415, "Format d'image non reconnu")

    result = _engine(request).inspect(image, category_hint=category or None)
    iid = str(uuid.uuid4())
    storage.save_artifacts(iid, image, result.anomaly_map_01, result.mask, result.regions)

    insp = Inspection(
        id=iid, part_reference=part_reference, original_filename=file.filename,
        width=int(image.shape[1]), height=int(image.shape[0]),
        category=result.category, category_confidence=result.category_confidence,
        category_source=result.category_source, engine=result.engine,
        model_version=result.model_version, calibration_source=result.calibration_source,
        anomaly_score=result.anomaly_score, confidence=result.confidence,
        verdict=result.verdict, status="pending_review",
        processing_ms=result.processing_ms, warnings=result.warnings,
    )
    insp.detections = [Detection(rank=k, **{f: getattr(r, f) for f in (
        "x", "y", "w", "h", "area_px", "max_score", "mean_score", "polygon",
        "defect_type", "defect_type_confidence", "source")}) for k, r in enumerate(result.regions, 1)]
    session.add(insp)
    session.commit()
    return _full(_get(session, iid))


@router.get("/inspections", response_model=Page)
def list_inspections(
    status: str | None = None,
    verdict: str | None = None,
    category: str | None = None,
    queue: bool = Query(False, description="tri par priorité de revue (non conformes d'abord)"),
    limit: int = Query(50, le=500),
    offset: int = 0,
    session: Session = Depends(get_session),
):
    q = select(Inspection).options(selectinload(Inspection.detections))
    if status:
        q = q.where(Inspection.status == status)
    if verdict:
        q = q.where(Inspection.verdict == verdict)
    if category:
        q = q.where(Inspection.category == category)
    total = session.scalar(select(func.count()).select_from(q.subquery()))
    items = list(session.scalars(q.order_by(Inspection.created_at.desc())))
    if queue:
        items.sort(key=lambda i: (VERDICT_PRIORITY.get(i.verdict, 9), -i.confidence))
    return Page(total=total or 0, items=[_summary(i) for i in items[offset:offset + limit]])


@router.get("/inspections/{inspection_id}", response_model=InspectionOut)
def get_inspection(inspection_id: str, session: Session = Depends(get_session)):
    return _full(_get(session, inspection_id))


@router.get("/inspections/{inspection_id}/image/{kind}")
def get_image(inspection_id: str, kind: str):
    try:
        path = storage.artifact_path(inspection_id, kind)
    except ValueError:
        raise HTTPException(404, "Type d'image inconnu")
    if not path.exists():
        raise HTTPException(404, "Image introuvable")
    return FileResponse(path, media_type="image/png")


@router.post("/inspections/{inspection_id}/reviews", response_model=InspectionOut, status_code=201)
def add_review(inspection_id: str, body: ReviewIn, session: Session = Depends(get_session)):
    insp = _get(session, inspection_id)
    valid_ids = {d.id for d in insp.detections}
    for fb in body.detection_feedback:
        if fb.detection_id not in valid_ids:
            raise HTTPException(422, f"Détection {fb.detection_id} n'appartient pas à cette inspection")
        if fb.corrected_type and fb.corrected_type not in GENERIC_DEFECTS:
            raise HTTPException(422, f"Type inconnu : {fb.corrected_type}")
    if body.decision == "conforme" and any(fb.is_true_defect for fb in body.detection_feedback):
        raise HTTPException(422, "Incohérent : pièce déclarée conforme avec un défaut confirmé")

    insp.reviews.append(Review(
        reviewer=body.reviewer, decision=body.decision, comment=body.comment,
        detection_feedback=[fb.model_dump() for fb in body.detection_feedback],
        added_regions=[r.model_dump() for r in body.added_regions],
        model_verdict_at_review=insp.verdict,
    ))
    insp.final_decision = body.decision
    insp.status = "needs_second_review" if body.decision == "inconclusif" else "reviewed"
    session.commit()
    return _full(_get(session, inspection_id))


# ---------------------------------------------------------------------- statistiques
@router.get("/stats")
def stats(session: Session = Depends(get_session)):
    insps = list(session.scalars(select(Inspection).options(
        selectinload(Inspection.detections), selectinload(Inspection.reviews))))
    by = lambda attr: {k: sum(1 for i in insps if getattr(i, attr) == k)  # noqa: E731
                       for k in sorted({getattr(i, attr) for i in insps if getattr(i, attr)})}

    # Concordance modèle / humain (pièces revues de manière concluante).
    # « Alerte » modèle = verdict « à vérifier » ou « non conforme probable ».
    reviewed = [i for i in insps if i.final_decision in ("conforme", "non_conforme")]
    tp = sum(1 for i in reviewed if i.final_decision == "non_conforme" and i.verdict != "conforme_probable")
    fn = sum(1 for i in reviewed if i.final_decision == "non_conforme" and i.verdict == "conforme_probable")
    fp = sum(1 for i in reviewed if i.final_decision == "conforme" and i.verdict != "conforme_probable")
    tn = sum(1 for i in reviewed if i.final_decision == "conforme" and i.verdict == "conforme_probable")

    det_true = det_false = missed = 0
    for i in insps:
        if i.reviews:
            last = i.reviews[-1]
            det_true += sum(1 for f in last.detection_feedback if f["is_true_defect"])
            det_false += sum(1 for f in last.detection_feedback if not f["is_true_defect"])
            missed += len(last.added_regions)

    return {
        "total": len(insps),
        "by_status": by("status"),
        "by_verdict": by("verdict"),
        "by_category": by("category"),
        "by_engine": by("engine"),
        "mean_processing_ms": round(sum(i.processing_ms for i in insps) / len(insps), 1) if insps else None,
        "human_vs_model": {
            "reviewed": len(reviewed),
            "defects_caught": tp, "defects_missed": fn,
            "false_alarms": fp, "correct_passes": tn,
            "miss_rate": fn / (tp + fn) if tp + fn else None,
        },
        "detection_feedback": {
            "confirmed": det_true, "rejected": det_false, "missed_regions_added": missed,
            "precision": det_true / (det_true + det_false) if det_true + det_false else None,
        },
    }


@router.get("/export/feedback.csv")
def export_feedback(session: Session = Depends(get_session)):
    """Export des revues humaines : matière première pour ré-entraîner (boucle active learning)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["inspection_id", "created_at", "category", "model_verdict", "confidence", "engine",
                "model_version", "reviewer", "decision", "detection_id", "x", "y", "w", "h",
                "predicted_type", "is_true_defect", "corrected_type", "comment"])
    q = select(Inspection).options(selectinload(Inspection.detections), selectinload(Inspection.reviews))
    for i in session.scalars(q):
        dets = {d.id: d for d in i.detections}
        for r in i.reviews:
            rows = r.detection_feedback or [None]
            for f in rows:
                d = dets.get(f["detection_id"]) if f else None
                w.writerow([i.id, r.created_at.isoformat(), i.category, r.model_verdict_at_review,
                            f"{i.confidence:.4f}", i.engine, i.model_version, r.reviewer, r.decision,
                            d.id if d else "", d.x if d else "", d.y if d else "", d.w if d else "",
                            d.h if d else "", d.defect_type if d else "",
                            f["is_true_defect"] if f else "", (f or {}).get("corrected_type") or "",
                            r.comment or ""])
            for a in r.added_regions:
                w.writerow([i.id, r.created_at.isoformat(), i.category, r.model_verdict_at_review,
                            f"{i.confidence:.4f}", i.engine, i.model_version, r.reviewer, r.decision,
                            "missed", a["x"], a["y"], a["w"], a["h"], "", True, a["defect_type"], ""])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=aerovision_feedback.csv"})
