import { createContext, useContext, useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import { DEMO, api } from "./api";
import type { SystemInfo } from "./api";
import InspectPage from "./pages/InspectPage";
import ListPage from "./pages/ListPage";
import DetailPage from "./pages/DetailPage";
import DashboardPage from "./pages/DashboardPage";

const SystemCtx = createContext<SystemInfo | null>(null);
export const useSystem = () => useContext(SystemCtx);

export default function App() {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.system().then(setSystem).catch((e) => setError(String(e)));
  }, []);

  return (
    <SystemCtx.Provider value={system}>
      <header className="topbar">
        <NavLink to="/inspect" className="brand" aria-label="AeroVision, accueil">
          <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
            <path d="M2 8V2h6M18 2h6v6M24 18v6h-6M8 24H2v-6" fill="none" stroke="#fff" strokeWidth="2.2" />
            <circle cx="13" cy="13" r="3.2" fill="#ff3b67" />
          </svg>
          <span className="brand-name">AeroVision</span>
        </NavLink>
        <nav>
          <NavLink to="/inspect">Inspecter</NavLink>
          <NavLink to="/queue">À valider</NavLink>
          <NavLink to="/history">Historique</NavLink>
          <NavLink to="/dashboard">Tableau de bord</NavLink>
        </nav>
        {system && (
          <div className="mode" title={system.mode === "patchcore" ? system.patchcore_categories.join(", ") : ""}>
            <span className={`led ${system.mode === "patchcore" ? "" : "off"}`} />
            {system.mode === "patchcore"
              ? `${system.patchcore_categories.length} modèle${system.patchcore_categories.length > 1 ? "s" : ""} PatchCore${system.classifier ? " et classifieur" : ""}`
              : "Mode OpenCV, sans modèle IA"}
          </div>
        )}
      </header>
      <div className="disclaimer">
        <strong>Aide à l'inspection, non qualifiée pour libérer une pièce aéronautique.</strong> La décision revient
        toujours à l'opérateur.
        {DEMO && (
          <>
            {" "}Démo statique : analyses calculées au préalable par les vrais modèles, sur des images MVTec AD
            (CC BY-NC-SA 4.0) jamais vues à l'entraînement. Vos validations restent dans ce navigateur.
          </>
        )}
      </div>
      {error && (
        <main style={{ paddingBottom: 0 }}>
          <div className="alert error">{DEMO ? "Données de démo introuvables" : "API injoignable"} : {error}</div>
        </main>
      )}
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/inspect" replace />} />
          <Route path="/inspect" element={<InspectPage />} />
          <Route path="/queue" element={<ListPage mode="queue" />} />
          <Route path="/history" element={<ListPage mode="history" />} />
          <Route path="/inspections/:id" element={<DetailPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </main>
    </SystemCtx.Provider>
  );
}
