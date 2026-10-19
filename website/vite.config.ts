import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The site is served as a GitHub Pages *project* page, so every asset URL is
// prefixed with the repository name: https://ma-lab-berkeley.github.io/Nestor/
// Override with VITE_BASE=/ for a custom domain or a local preview at the root.
export default defineConfig({
  base: process.env.VITE_BASE ?? "/Nestor/",
  plugins: [react()],
  build: {
    target: "es2022",
    sourcemap: false,
  },
});
