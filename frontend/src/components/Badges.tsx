import { DECISION_LABEL, STATUS_LABEL, VERDICT_LABEL, pct } from "../api";
import type { Decision, Verdict } from "../api";

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return <span className={`badge v-${verdict}`}>{VERDICT_LABEL[verdict] ?? verdict}</span>;
}

export function StatusBadge({ status, decision }: { status: string; decision?: Decision | null }) {
  if (decision && status === "reviewed")
    return <span className={`badge d-${decision}`}>{DECISION_LABEL[decision]}</span>;
  return <span className={`badge s-${status}`}>{STATUS_LABEL[status] ?? status}</span>;
}

export function EngineBadge({ engine }: { engine: string }) {
  return (
    <span className={`badge e-${engine}`} title={engine === "baseline_opencv" ? "Vision classique, sans IA" : "Deep learning"}>
      {engine === "patchcore" ? "PatchCore" : engine === "baseline_opencv" ? "OpenCV (sans IA)" : engine}
    </span>
  );
}

/** Jauge go / no-go : probabilité calibrée de défaut, avec les zones de la politique de décision. */
export function ConfidenceGauge({ value, low, high }: { value: number; low: number; high: number }) {
  return (
    <div className="gauge" title="Probabilité calibrée que la pièce soit défectueuse">
      <div className="gauge-value">
        <strong>{pct(value, value > 0.995 || value < 0.005 ? 0 : 1)}</strong>
        <span>probabilité de défaut</span>
      </div>
      <div className="gauge-track" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={value}
           aria-label="Probabilité de défaut">
        <div className="z ok" style={{ width: `${low * 100}%` }} />
        <div className="z mid" style={{ width: `${(high - low) * 100}%` }} />
        <div className="z ko" style={{ width: `${(1 - high) * 100}%` }} />
        <div className="gauge-needle" style={{ left: `${Math.min(99.5, Math.max(0.5, value * 100))}%` }} />
      </div>
      <div className="gauge-scale">
        <span style={{ width: `${low * 100}%` }}>Conforme</span>
        <span style={{ width: `${(high - low) * 100}%` }}>À vérifier</span>
        <span style={{ width: `${(1 - high) * 100}%` }}>Non conforme</span>
      </div>
    </div>
  );
}
