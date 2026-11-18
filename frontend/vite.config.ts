import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// mode "demo" : build statique pour GitHub Pages (chemins relatifs, aucun backend)
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === "demo" ? "./" : "/",
  server: {
    port: 5173,
    proxy: { "/api": "http://localhost:8000" },
  },
}));
