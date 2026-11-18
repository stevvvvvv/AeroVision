import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { DECISION_LABEL, VERDICT_LABEL, api, partName, pct } from "../api";
import type { Decision, Inspection } from "../api";
import { useSystem } from "../App";
import { ConfidenceGauge, EngineBadge, StatusBadge, VerdictBadge } from "../components/Badges";
import ImageViewer from "../components/ImageViewer";
import type { Box } from "../components/ImageViewer";

type View = "original" | "overlay" | "heatmap" | "mask";

export default function DetailPage() {
  const { id } = useParams<{ id: string }>();
  const system = useSystem();
  const labels = system?.defect_labels_fr ?? {};
  const [insp, setInsp] = useState<Inspection | null>(null);
  const [view, setView] = useState<View>("original");
  const [selected, setSelected] = useState<number | null>(null);

  // état du formulaire de revue
  const [feedback, setFeedback] = useState<Record<number, boolean | undefined>>({});
  const [corrected, setCorrected] = useState<Record<number, string>>({});
  const [added, setAdded] = useState<Box[]>([]);
  const [drawMode, setDrawMode] = useState(false);
  const [reviewer, setReviewer] = useState(() => localStorageGet("aerovision.reviewer"));
  const [decision, setDecision] = useState<Decision | "">("");
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (id) api.get(id).then(setInsp).catch((e) => setError(String(e)));
  }, [id]);

  if (!insp) return <section className="card">{error ?? "Chargement…"}</section>;

  const policy = system?.decision_policy ?? { review_low: 0.2, review_high: 0.8 };
  const allEvaluated = insp.detections.every((d) => feedback[d.id] !== undefined);
  const anyConfirmed = Object.values(feedback).some((v) => v === true) || added.length > 0;

  const submit = async () => {
    if (!decision) return;
    setSaving(true);
    setError(null);
    try {
      localStorageSet("aerovision.reviewer", reviewer);
      const r = await api.review(insp.id, {
        reviewer,
        decision,
        comment: comment || undefined,
        detection_feedback: insp.detections
          .filter((d) => feedback[d.id] !== undefined)
          .map((d) => ({
            detection_id: d.id,
            is_true_defect: feedback[d.id]!,
            corrected_type: feedback[d.id] ? corrected[d.id] || null : null,
          })),
        added_regions: added,
      });
      setInsp(r);
      setFeedback({});
      setCorrected({});
      setAdded([]);
      setDecision("");
      setComment("");
    } catch (e) {
      setError(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="detail">
      <section className="card viewer-card">
        <div className="viewer-toolbar">
          <div className="tabs" role="tablist" aria-label="Vue de l'image">
            {(["original", "overlay", "heatmap", "mask"] as View[]).map((v) => (
              <button key={v} className={view === v ? "active" : ""} onClick={() => setView(v)}>
                {{ original: "Zones", overlay: "Superposition", heatmap: "Carte d'anomalie", mask: "Segmentation" }[v]}
              </button>
            ))}
          </div>
          {insp.status !== "reviewed" && (
            <button className={`tool ${drawMode ? "active" : ""}`} onClick={() => { setDrawMode(!drawMode); setView("original"); }}
                    title="Tracer une zone que le modèle n'a pas détectée">
              {drawMode ? "Terminer le tracé" : "Signaler un défaut manqué"}
            </button>
          )}
        </div>
        <ImageViewer
          src={api.imageUrl(insp.id, view)}
          width={insp.width}
          height={insp.height}
          detections={insp.detections}
          selected={selected}
          onSelect={setSelected}
          feedback={feedback}
          addedBoxes={added}
          drawMode={drawMode}
          onDraw={(b) => setAdded([...added, { ...b, defect_type: "other" }])}
          showBoxes={view === "original" || view === "mask"}
        />
        <p className="hint">
          {drawMode
            ? "Tracez un rectangle autour du défaut que le modèle n'a pas vu."
            : `Analysé en ${insp.processing_ms} ms par ${insp.model_version}, image de ${insp.width} × ${insp.height} px.`}
        </p>
      </section>

      <aside className="side">
        <section className="card">
          <div className="verdict-head">
            <div>
              <h2>{partName(insp.category)}</h2>
              <div className="ref">{insp.part_reference ?? insp.original_filename}</div>
            </div>
            <StatusBadge status={insp.status} decision={insp.final_decision} />
          </div>
          <VerdictBadge verdict={insp.verdict} />
          <ConfidenceGauge value={insp.confidence} low={policy.review_low} high={policy.review_high} />
          {insp.warnings.map((w) => (
            <div key={w} className="alert warn">{w}</div>
          ))}
          <dl className="kv">
            {insp.ground_truth && (
              <>
                <dt>Vérité terrain</dt>
                <dd>
                  {insp.ground_truth.is_anomalous ? `Défaut « ${insp.ground_truth.defect} »` : "Pièce saine"}
                  <div className="muted small">Étiquette MVTec, invisible pour l'opérateur en production.</div>
                </dd>
              </>
            )}
            <dt>Identification</dt>
            <dd>
              {insp.category_source === "classifier" && insp.category_confidence !== null
                ? `Classifieur CNN, ${pct(insp.category_confidence)} de confiance`
                : insp.category_source === "user" ? "Saisie par l'opérateur" : "Modèle unique"}
            </dd>
            <dt>Moteur</dt>
            <dd><EngineBadge engine={insp.engine} /></dd>
            <dt>Score brut</dt>
            <dd>{insp.anomaly_score.toFixed(3)} <span className="muted small">(distance au plus proche patch sain)</span></dd>
          </dl>
        </section>

        <section className="card">
          <h2>{insp.detections.length === 0 ? "Aucune zone suspecte" : insp.detections.length === 1 ? "1 zone suspecte" : `${insp.detections.length} zones suspectes`}</h2>
          {insp.detections.length === 0 && (
            <p className="muted small">Aucune région ne dépasse le seuil de segmentation. Si vous voyez un défaut, signalez-le sur l'image.</p>
          )}
          <ol className="dets">
            {insp.detections.map((d) => (
              <li key={d.id} className={`${selected === d.id ? "sel" : ""} ${feedback[d.id] === false ? "rejected" : ""}`}
                  onClick={() => setSelected(d.id)}>
                <div className="row between">
                  <strong>Zone {d.rank} : {(labels[d.defect_type] ?? d.defect_type).toLowerCase()}</strong>
                  <span className="muted small">{d.area_px} px²</span>
                </div>
                <div className="muted small">
                  Score maximal {d.max_score.toFixed(2)}
                  {d.defect_type_confidence !== null && `, type identifié à ${pct(d.defect_type_confidence)}`}
                </div>
                {insp.aero_hints[d.defect_type] && <div className="hint small">{insp.aero_hints[d.defect_type]}</div>}
                {insp.status !== "reviewed" && (
                  <div className="row gap fb">
                    <button className={feedback[d.id] === true ? "danger" : ""}
                            onClick={(e) => { e.stopPropagation(); setFeedback({ ...feedback, [d.id]: true }); }}>
                      Défaut réel
                    </button>
                    <button className={feedback[d.id] === false ? "active" : ""}
                            onClick={(e) => { e.stopPropagation(); setFeedback({ ...feedback, [d.id]: false }); }}>
                      Fausse alarme
                    </button>
                    {feedback[d.id] === true && (
                      <select value={corrected[d.id] ?? ""} onClick={(e) => e.stopPropagation()}
                              onChange={(e) => setCorrected({ ...corrected, [d.id]: e.target.value })}>
                        <option value="">Type : {labels[d.defect_type] ?? d.defect_type}</option>
                        {(system?.defect_classes ?? []).map((c) => (
                          <option key={c} value={c}>{labels[c] ?? c}</option>
                        ))}
                      </select>
                    )}
                  </div>
                )}
              </li>
            ))}
          </ol>
          {added.length > 0 && (
            <>
              <h3>Défauts ajoutés par l'opérateur</h3>
              <ul className="dets">
                {added.map((b, i) => (
                  <li key={i} className="added">
                    <div className="row between">
                      <select value={b.defect_type}
                              onChange={(e) => setAdded(added.map((x, j) => (j === i ? { ...x, defect_type: e.target.value } : x)))}>
                        {(system?.defect_classes ?? []).map((c) => (
                          <option key={c} value={c}>{labels[c] ?? c}</option>
                        ))}
                      </select>
                      <button onClick={() => setAdded(added.filter((_, j) => j !== i))}>Retirer</button>
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>

        {insp.status !== "reviewed" && (
          <section className="card">
            <h2>Décision</h2>
            <label>
              Opérateur
              <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Nom / matricule" />
            </label>
            <div className="decision">
              {(["conforme", "non_conforme", "inconclusif"] as Decision[]).map((d) => (
                <label key={d} className={`radio d-${d} ${decision === d ? "on" : ""}`}>
                  <input type="radio" name="decision" checked={decision === d} onChange={() => setDecision(d)} />
                  {DECISION_LABEL[d]}
                </label>
              ))}
            </div>
            {decision === "conforme" && anyConfirmed && (
              <div className="alert error">Incohérent : un défaut est confirmé.</div>
            )}
            {!allEvaluated && insp.detections.length > 0 && (
              <div className="alert warn">Toutes les zones n'ont pas été évaluées.</div>
            )}
            <label>
              Commentaire
              <textarea value={comment} onChange={(e) => setComment(e.target.value)} rows={3}
                        placeholder="Observations, mesures, renvoi vers CND…" />
            </label>
            {error && <div className="alert error">{error}</div>}
            <button className="primary" onClick={submit}
                    disabled={!decision || reviewer.trim().length < 2 || saving || (decision === "conforme" && anyConfirmed)}>
              {saving ? "Enregistrement…" : "Signer la décision"}
            </button>
          </section>
        )}

        {insp.reviews.length > 0 && (
          <section className="card">
            <h2>Historique des décisions</h2>
            <ul className="audit">
              {insp.reviews.map((r) => (
                <li key={r.id}>
                  <div className="row between">
                    <strong>{DECISION_LABEL[r.decision]}</strong>
                    <span className="muted small">{new Date(r.created_at).toLocaleString("fr-FR")}</span>
                  </div>
                  <div className="small">
                    Signée par {r.reviewer}. Le modèle proposait « {VERDICT_LABEL[r.model_verdict_at_review].toLowerCase()} ».{" "}
                    Zones : {r.detection_feedback.filter((f) => f.is_true_defect).length} confirmée(s),{" "}
                    {r.detection_feedback.filter((f) => !f.is_true_defect).length} rejetée(s),{" "}
                    {r.added_regions.length} ajoutée(s).
                  </div>
                  {r.comment && <div className="small muted">« {r.comment} »</div>}
                </li>
              ))}
            </ul>
          </section>
        )}
      </aside>
    </div>
  );
}

function localStorageGet(k: string): string {
  try {
    return localStorage.getItem(k) ?? "";
  } catch {
    return "";
  }
}
function localStorageSet(k: string, v: string) {
  try {
    localStorage.setItem(k, v);
  } catch {
    /* stockage indisponible */
  }
}
