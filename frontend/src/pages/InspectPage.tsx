import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { DEMO, PART_LABEL, api, partName } from "../api";
import type { Inspection } from "../api";
import { useSystem } from "../App";

export default function InspectPage() {
  return DEMO ? <DemoInspect /> : <LiveInspect />;
}

/** Démo statique : on choisit une image d'évaluation MVTec déjà passée dans le modèle. */
function DemoInspect() {
  const nav = useNavigate();
  const [samples, setSamples] = useState<Inspection[]>([]);
  const [selected, setSelected] = useState<Inspection | null>(null);

  useEffect(() => {
    api.samples().then(setSamples);
  }, []);

  const byCat: Record<string, Inspection[]> = {};
  samples.forEach((s) => (byCat[s.ground_truth?.category ?? s.category] ??= []).push(s));

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Choisir une pièce à inspecter</h1>
          <p>
            Ces images n'ont jamais servi à l'entraînement ni au réglage des seuils. Chacune a été analysée par le
            pipeline réel : identification de la pièce, détection d'anomalie, localisation des zones.
          </p>
        </div>
      </div>
      <section className="card">
        {Object.entries(byCat).map(([cat, items]) => (
          <div key={cat} className="part-group">
            <div className="part-group-head">
              <h2>{partName(cat)}</h2>
              <span>{PART_LABEL[cat]?.proxy} · {items.length} images</span>
            </div>
            <div className="gallery">
              {items.map((s) => (
                <button key={s.id} className={`sample ${selected?.id === s.id ? "on" : ""}`}
                        onClick={() => setSelected(s)} onDoubleClick={() => nav(`/inspections/${s.id}`)}
                        aria-label={`${partName(cat)}, image ${s.original_filename}`}>
                  <img src={api.imageUrl(s.id, "original")} alt="" loading="lazy" />
                </button>
              ))}
            </div>
          </div>
        ))}
        <div className="sticky-cta">
          <span className="muted">
            {selected
              ? `${partName(selected.ground_truth?.category ?? selected.category)}, image ${selected.original_filename}`
              : "Sélectionnez une image."}
          </span>
          <button className="primary" disabled={!selected} onClick={() => selected && nav(`/inspections/${selected.id}`)}>
            Inspecter cette pièce
          </button>
        </div>
      </section>
    </>
  );
}

function LiveInspect() {
  const system = useSystem();
  const nav = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [partRef, setPartRef] = useState("");
  const [category, setCategory] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pick = (f: File | null) => {
    setFile(f);
    setError(null);
    setPreview(f ? URL.createObjectURL(f) : null);
  };

  const submit = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.inspect(file, partRef || undefined, category || undefined);
      nav(`/inspections/${r.id}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const categories = system?.patchcore_categories ?? [];

  return (
    <section className="card narrow">
      <h1>Nouvelle inspection</h1>
      <p className="muted">Photographie de la pièce, fond neutre, éclairage proche de celui de l'entraînement.</p>
      <label
        className={`dropzone ${preview ? "has" : ""}`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          pick(e.dataTransfer.files?.[0] ?? null);
        }}
      >
        <input type="file" accept="image/*" onChange={(e) => pick(e.target.files?.[0] ?? null)} hidden />
        {preview ? <img src={preview} alt="aperçu" /> : <span>Déposez une image ou cliquez pour la choisir</span>}
      </label>

      <div className="grid2">
        <label>
          Référence pièce / n° de série
          <input value={partRef} onChange={(e) => setPartRef(e.target.value)} placeholder="ex. PN-4471-A / SN 00213" />
        </label>
        <label>
          Type de pièce
          <select value={category} onChange={(e) => setCategory(e.target.value)}>
            <option value="">{system?.classifier ? "Identification automatique (CNN)" : "Automatique"}</option>
            {categories.map((c) => (
              <option key={c} value={c}>{partName(c)}</option>
            ))}
          </select>
        </label>
      </div>

      {error && <div className="alert error">{error}</div>}
      <button className="primary" disabled={!file || busy} onClick={submit}>
        {busy ? "Analyse en cours…" : "Lancer l'inspection"}
      </button>
    </section>
  );
}
