import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, partName, pct } from "../api";
import type { InspectionSummary } from "../api";
import { EngineBadge, StatusBadge, VerdictBadge } from "../components/Badges";

export default function ListPage({ mode }: { mode: "queue" | "history" }) {
  const [items, setItems] = useState<InspectionSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [verdict, setVerdict] = useState("");
  const [status, setStatus] = useState("");

  useEffect(() => {
    const params =
      mode === "queue"
        ? { status: "pending_review", queue: true, verdict, limit: 200 }
        : { status, verdict, limit: 200 };
    api.list(params).then((p) => {
      setItems(p.items);
      setTotal(p.total);
    });
  }, [mode, verdict, status]);

  return (
    <>
    <div className="page-head">
      <div>
        <h1>{mode === "queue" ? "À valider" : "Historique"}</h1>
        <p>
          {mode === "queue"
            ? "Les pièces jugées non conformes par le modèle passent en premier, puis par probabilité de défaut décroissante."
            : `${total} inspection(s), de la plus récente à la plus ancienne.`}
        </p>
      </div>
        <div className="row gap">
          {mode === "history" && (
            <select value={status} onChange={(e) => setStatus(e.target.value)}>
              <option value="">Tous statuts</option>
              <option value="pending_review">En attente</option>
              <option value="reviewed">Revues</option>
              <option value="needs_second_review">2e avis requis</option>
            </select>
          )}
          <select value={verdict} onChange={(e) => setVerdict(e.target.value)}>
            <option value="">Tous verdicts modèle</option>
            <option value="non_conforme_probable">Non conforme probable</option>
            <option value="a_verifier">À vérifier</option>
            <option value="conforme_probable">Conforme probable</option>
          </select>
        </div>
    </div>
    <section className="card">
      <table>
        <thead>
          <tr>
            <th></th><th>Date</th><th>Référence</th><th>Pièce</th><th>Moteur</th>
            <th>Verdict du modèle</th><th className="num">Probabilité de défaut</th><th className="num">Zones</th><th>Statut</th>
          </tr>
        </thead>
        <tbody>
          {items.map((i) => (
            <tr key={i.id}>
              <td><Link to={`/inspections/${i.id}`}><img className="thumb" src={api.imageUrl(i.id, "overlay")} alt="" /></Link></td>
              <td><Link to={`/inspections/${i.id}`}>{new Date(i.created_at).toLocaleString("fr-FR")}</Link></td>
              <td>{i.part_reference ?? "—"}</td>
              <td>{partName(i.category)}</td>
              <td><EngineBadge engine={i.engine} /></td>
              <td><VerdictBadge verdict={i.verdict} /></td>
              <td className="num">{pct(i.confidence, 1)}</td>
              <td className="num">{i.n_detections}</td>
              <td><StatusBadge status={i.status} decision={i.final_decision} /></td>
            </tr>
          ))}
          {items.length === 0 && (
            <tr><td colSpan={9} className="empty">{mode === "queue" ? "Rien à valider : toutes les inspections ont été revues." : "Aucune inspection pour ces filtres."}</td></tr>
          )}
        </tbody>
      </table>
    </section>
    </>
  );
}
