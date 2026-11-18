export type Verdict = "conforme_probable" | "a_verifier" | "non_conforme_probable";
export type Decision = "conforme" | "non_conforme" | "inconclusif";

export interface Detection {
  id: number;
  rank: number;
  x: number;
  y: number;
  w: number;
  h: number;
  area_px: number;
  max_score: number;
  mean_score: number;
  polygon: number[][];
  defect_type: string;
  defect_type_confidence: number | null;
  source: string;
}

export interface Review {
  id: number;
  created_at: string;
  reviewer: string;
  decision: Decision;
  comment: string | null;
  detection_feedback: { detection_id: number; is_true_defect: boolean; corrected_type: string | null }[];
  added_regions: { x: number; y: number; w: number; h: number; defect_type: string }[];
  model_verdict_at_review: Verdict;
}

export interface InspectionSummary {
  id: string;
  created_at: string;
  part_reference: string | null;
  original_filename: string | null;
  category: string;
  engine: string;
  anomaly_score: number;
  confidence: number;
  verdict: Verdict;
  status: string;
  final_decision: Decision | null;
  n_detections: number;
}

export interface Inspection extends InspectionSummary {
  width: number;
  height: number;
  category_confidence: number | null;
  category_source: string;
  model_version: string;
  calibration_source: string;
  processing_ms: number;
  warnings: string[];
  detections: Detection[];
  reviews: Review[];
  aero_hints: Record<string, string>;
  /** démo statique uniquement : étiquette MVTec de l'image */
  ground_truth?: { category: string; defect: string; is_anomalous: boolean; generic_type: string | null };
}

export interface SystemInfo {
  mode: string;
  patchcore_categories: string[];
  classifier: { classes: string[]; test_accuracy?: number } | null;
  yolo: boolean;
  defect_classes: string[];
  defect_labels_fr: Record<string, string>;
  decision_policy: { review_low: number; review_high: number };
  disclaimer: string;
  patchcore_reports?: Record<string, ModelReport | null>;
  demo?: boolean;
  generated_at?: string;
}

export interface ModelReport {
  split: { fit: number; normal_holdout: number; calibration: number; evaluation: number };
  evaluation_image: {
    image_auroc?: number;
    recall?: number;
    false_negative_rate?: number;
    false_positive_rate?: number;
    confusion?: { tp: number; fn: number; fp: number; tn: number };
  };
  evaluation_pixel: { pixel_auroc?: number };
}

export interface ReviewPayload {
  reviewer: string;
  decision: Decision;
  comment?: string;
  detection_feedback: { detection_id: number; is_true_defect: boolean; corrected_type?: string | null }[];
  added_regions: { x: number; y: number; w: number; h: number; defect_type: string }[];
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, init);
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      const body = await r.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* corps non JSON */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

const liveApi = {
  system: () => req<SystemInfo>("/system"),
  stats: () => req<any>("/stats"),
  list: (params: ListParams) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => v !== undefined && v !== "" && q.set(k, String(v)));
    return req<{ total: number; items: InspectionSummary[] }>(`/inspections?${q}`);
  },
  get: (id: string) => req<Inspection>(`/inspections/${id}`),
  inspect: (file: File, partReference?: string, category?: string) => {
    const fd = new FormData();
    fd.append("file", file);
    if (partReference) fd.append("part_reference", partReference);
    if (category) fd.append("category", category);
    return req<Inspection>("/inspections", { method: "POST", body: fd });
  },
  review: (id: string, payload: ReviewPayload) =>
    req<Inspection>(`/inspections/${id}/reviews`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  imageUrl: (id: string, kind: ImageKind) => `/api/inspections/${id}/image/${kind}`,
  samples: async (): Promise<Inspection[]> => [],
};

type ImageKind = "original" | "heatmap" | "overlay" | "mask";
type ListParams = Record<string, string | boolean | number | undefined>;

// ---------------------------------------------------------------------------
// Démo statique (GitHub Pages) : résultats pré-calculés par les vrais modèles,
// revues opérateur conservées dans le navigateur (localStorage).
// ---------------------------------------------------------------------------
export const DEMO = import.meta.env.MODE === "demo";
const BASE = `${import.meta.env.BASE_URL}demo/`;
const REVIEWS_KEY = "aerovision.demo.reviews";
let cache: { system: SystemInfo; inspections: Inspection[] } | null = null;

async function load() {
  if (!cache) {
    const [system, inspections] = await Promise.all([
      fetch(`${BASE}system.json`).then((r) => r.json()),
      fetch(`${BASE}inspections.json`).then((r) => r.json()),
    ]);
    cache = { system, inspections };
  }
  return cache;
}

function storedReviews(): Record<string, Review[]> {
  try {
    return JSON.parse(localStorage.getItem(REVIEWS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function withReviews(i: Inspection): Inspection {
  const reviews = storedReviews()[i.id] ?? [];
  if (!reviews.length) return { ...i, reviews: [] };
  const last = reviews[reviews.length - 1];
  return {
    ...i,
    reviews,
    final_decision: last.decision,
    status: last.decision === "inconclusif" ? "needs_second_review" : "reviewed",
  };
}

const PRIORITY: Record<string, number> = { non_conforme_probable: 0, a_verifier: 1, conforme_probable: 2 };

const demoApi: typeof liveApi = {
  system: async () => (await load()).system,
  list: async (params: ListParams) => {
    let items = (await load()).inspections.map(withReviews);
    if (params.status) items = items.filter((i) => i.status === params.status);
    if (params.verdict) items = items.filter((i) => i.verdict === params.verdict);
    if (params.queue) items.sort((a, b) => PRIORITY[a.verdict] - PRIORITY[b.verdict] || b.confidence - a.confidence);
    return { total: items.length, items };
  },
  get: async (id: string) => {
    const i = (await load()).inspections.find((x) => x.id === id);
    if (!i) throw new Error("Inspection introuvable");
    return withReviews(i);
  },
  inspect: async () => {
    throw new Error("Démo statique : l'envoi d'images nécessite le backend (voir le README).");
  },
  review: async (id: string, payload: ReviewPayload) => {
    const insp = await demoApi.get(id);
    if (payload.decision === "conforme" && payload.detection_feedback.some((f) => f.is_true_defect))
      throw new Error("Incohérent : pièce déclarée conforme avec un défaut confirmé");
    const all = storedReviews();
    const review: Review = {
      id: Date.now(),
      created_at: new Date().toISOString(),
      reviewer: payload.reviewer,
      decision: payload.decision,
      comment: payload.comment ?? null,
      detection_feedback: payload.detection_feedback.map((f) => ({ ...f, corrected_type: f.corrected_type ?? null })),
      added_regions: payload.added_regions,
      model_verdict_at_review: insp.verdict,
    };
    all[id] = [...(all[id] ?? []), review];
    try {
      localStorage.setItem(REVIEWS_KEY, JSON.stringify(all));
    } catch {
      /* stockage indisponible : la revue ne survivra pas au rechargement */
    }
    return demoApi.get(id);
  },
  stats: async () => computeStats((await load()).inspections.map(withReviews)),
  imageUrl: (id: string, kind: ImageKind) => `${BASE}${id}/${kind}.${kind === "mask" ? "png" : "jpg"}`,
  samples: async () => (await load()).inspections,
};

/** Mêmes indicateurs que GET /api/stats côté backend. */
function computeStats(insps: Inspection[]) {
  const by = (f: (i: Inspection) => string | null) => {
    const out: Record<string, number> = {};
    insps.forEach((i) => {
      const k = f(i);
      if (k) out[k] = (out[k] ?? 0) + 1;
    });
    return out;
  };
  const reviewed = insps.filter((i) => i.final_decision === "conforme" || i.final_decision === "non_conforme");
  const alert = (i: Inspection) => i.verdict !== "conforme_probable";
  const tp = reviewed.filter((i) => i.final_decision === "non_conforme" && alert(i)).length;
  const fn = reviewed.filter((i) => i.final_decision === "non_conforme" && !alert(i)).length;
  const fp = reviewed.filter((i) => i.final_decision === "conforme" && alert(i)).length;
  const tn = reviewed.filter((i) => i.final_decision === "conforme" && !alert(i)).length;
  let conf = 0, rej = 0, missed = 0;
  insps.forEach((i) => {
    const last = i.reviews[i.reviews.length - 1];
    if (!last) return;
    conf += last.detection_feedback.filter((f) => f.is_true_defect).length;
    rej += last.detection_feedback.filter((f) => !f.is_true_defect).length;
    missed += last.added_regions.length;
  });
  return {
    total: insps.length,
    by_status: by((i) => i.status),
    by_verdict: by((i) => i.verdict),
    by_category: by((i) => i.category),
    by_engine: by((i) => i.engine),
    mean_processing_ms: insps.length
      ? Math.round((insps.reduce((s, i) => s + i.processing_ms, 0) / insps.length) * 10) / 10
      : null,
    human_vs_model: {
      reviewed: reviewed.length, defects_caught: tp, defects_missed: fn, false_alarms: fp, correct_passes: tn,
      miss_rate: tp + fn ? fn / (tp + fn) : null,
    },
    detection_feedback: {
      confirmed: conf, rejected: rej, missed_regions_added: missed,
      precision: conf + rej ? conf / (conf + rej) : null,
    },
  };
}

export const api = DEMO ? demoApi : liveApi;

export const VERDICT_LABEL: Record<Verdict, string> = {
  conforme_probable: "Conforme probable",
  a_verifier: "À vérifier",
  non_conforme_probable: "Non conforme probable",
};

export const DECISION_LABEL: Record<Decision, string> = {
  conforme: "Conforme",
  non_conforme: "Non conforme",
  inconclusif: "Inconclusif",
};

export const STATUS_LABEL: Record<string, string> = {
  pending_review: "En attente de revue",
  reviewed: "Revue",
  needs_second_review: "2e avis requis",
};

export const pct = (x: number | null | undefined, d = 0) =>
  x === null || x === undefined ? "—" : `${(x * 100).toFixed(d)} %`;

/** Nom lisible des pièces et pièce aéronautique dont elles sont le proxy. */
export const PART_LABEL: Record<string, { name: string; proxy: string }> = {
  metal_nut: { name: "Écrou usiné", proxy: "proxy d'une fixation métallique" },
  screw: { name: "Vis", proxy: "proxy de la visserie" },
  cable: { name: "Câble", proxy: "proxy d'un harnais électrique" },
  grid: { name: "Grille métallique", proxy: "proxy d'une structure ajourée" },
  transistor: { name: "Transistor", proxy: "proxy d'un composant avionique" },
  synthetic_plate: { name: "Plaque synthétique", proxy: "données de test" },
};
export const partName = (c: string) => PART_LABEL[c]?.name ?? c;
