import { useEffect, useState } from "react";
import { STATUS_LABEL, VERDICT_LABEL, api, partName, pct } from "../api";
import type { Verdict } from "../api";
import { useSystem } from "../App";

export default function DashboardPage() {
  const [s, setS] = useState<any>(null);
  const system = useSystem();
  useEffect(() => {
    api.stats().then(setS);
  }, []);
  if (!s) return <section className="card">Chargement…</section>;

  const hm = s.human_vs_model;
  const fb = s.detection_feedback;

  return (
    <div className="dash">
      <div className="page-head">
        <div>
          <h1>Tableau de bord qualité</h1>
          <p>Ce que le modèle détecte, ce que les opérateurs confirment, et surtout ce qui a échappé au modèle.</p>
        </div>
      </div>
      <section className="kpis">
        <Kpi label="Défauts manqués par le modèle" value={hm.miss_rate === null ? "—" : pct(hm.miss_rate, 1)}
             note={hm.defects_caught + hm.defects_missed
               ? `${hm.defects_missed} sur ${hm.defects_caught + hm.defects_missed} pièces jugées non conformes par l'opérateur`
               : "Validez des pièces pour alimenter cet indicateur."} critical />
        <Kpi label="Inspections" value={s.total} />
        <Kpi label="À valider" value={s.by_status.pending_review ?? 0} />
        <Kpi label="Zones confirmées" value={fb.precision === null ? "—" : pct(fb.precision, 0)}
             note={`${fb.confirmed} confirmées, ${fb.rejected} fausses alarmes`} />
        <Kpi label="Temps d'analyse moyen" value={s.mean_processing_ms ? `${Math.round(s.mean_processing_ms)} ms` : "—"} />
      </section>

      {system?.patchcore_reports && <ModelsCard reports={system.patchcore_reports} classifierAcc={system.classifier?.test_accuracy} />}

      <section className="card">
        <h2>Modèle contre opérateur</h2>
        <p className="hint">Sur {hm.reviewed} pièce(s) revue(s) avec une décision conforme / non conforme.</p>
        <table className="confusion">
          <thead>
            <tr><th></th><th>Opérateur : non conforme</th><th>Opérateur : conforme</th></tr>
          </thead>
          <tbody>
            <tr><th>Modèle : alerte (à vérifier ou non conforme)</th><td className="good">{hm.defects_caught}</td><td>{hm.false_alarms}</td></tr>
            <tr><th>Modèle : conforme probable</th><td className="bad">{hm.defects_missed}</td><td className="good">{hm.correct_passes}</td></tr>
          </tbody>
        </table>
        <p className="hint">La case rouge est l'erreur qui compte en aéronautique : un défaut réel que le modèle a laissé passer.</p>
      </section>

      <div className="grid2">
        <Bars title="Verdicts du modèle" data={s.by_verdict} labels={VERDICT_LABEL as Record<Verdict, string>} />
        <Bars title="Statut des revues" data={s.by_status} labels={STATUS_LABEL} />
        <Bars title="Par type de pièce" data={Object.fromEntries(Object.entries(s.by_category as Record<string, number>).map(([k, v]) => [partName(k), v]))} />
        <Bars title="Par moteur d'inspection" data={s.by_engine} labels={{ patchcore: "PatchCore", baseline_opencv: "OpenCV, sans IA" }} />
      </div>
    </div>
  );
}

function Kpi({ label, value, note, critical }: { label: string; value: any; note?: string; critical?: boolean }) {
  return (
    <div className={`kpi ${critical ? "critical" : ""}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {note && <div className="small muted">{note}</div>}
    </div>
  );
}

function Bars({ title, data, labels = {} }: { title: string; data: Record<string, number>; labels?: Record<string, string> }) {
  const max = Math.max(1, ...Object.values(data));
  return (
    <section className="card">
      <h2>{title}</h2>
      {Object.keys(data).length === 0 && <p className="muted">Aucune donnée.</p>}
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="bar">
          <span className="bar-label">{labels[k] ?? k}</span>
          <span className="bar-track"><span className="bar-fill" style={{ width: `${(v / max) * 100}%` }} /></span>
          <span className="bar-val">{v}</span>
        </div>
      ))}
    </section>
  );
}

function ModelsCard({ reports, classifierAcc }: { reports: Record<string, any>; classifierAcc?: number }) {
  const rows = Object.entries(reports).filter(([, r]) => r);
  if (!rows.length) return null;
  return (
    <section className="card">
      <h2>Performances mesurées des modèles</h2>
      <p className="hint">Sur un jeu d'évaluation jamais vu à l'entraînement ni au réglage des seuils.</p>
      <table>
        <thead>
          <tr>
            <th>Pièce</th><th className="num">AUROC image</th><th className="num">AUROC pixel</th><th className="num">Défauts manqués</th><th className="num">Fausses alarmes</th><th className="num">Images évaluées</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([cat, r]) => {
            const c = r.evaluation_image.confusion;
            return (
              <tr key={cat}>
                <td>{partName(cat)}</td>
                <td className="num">{r.evaluation_image.image_auroc?.toFixed(3) ?? "—"}</td>
                <td className="num">{r.evaluation_pixel.pixel_auroc?.toFixed(3) ?? "—"}</td>
                <td className="num">{c ? `${c.fn} / ${c.fn + c.tp}` : "—"}</td>
                <td className="num">{c ? `${c.fp} / ${c.fp + c.tn}` : "—"}</td>
                <td className="num">{r.split.evaluation}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="hint">
        Seuils réglés pour maximiser le F1. En aéronautique, on les déplacerait pour réduire les défauts manqués,
        au prix de davantage de fausses alarmes.{classifierAcc !== undefined && ` Identification des pièces : ${pct(classifierAcc, 1)} sur le test.`}
      </p>
    </section>
  );
}
