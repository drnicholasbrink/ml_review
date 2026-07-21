import { defineConfig } from "vite";

export default defineConfig({
  base: "/static/atlas/",
  build: {
    outDir: "ml_review_app/static/atlas",
    emptyOutDir: true,
    sourcemap: false,
    rollupOptions: {
      input: "frontend/atlas/main.js",
      output: {
        entryFileNames: "atlas.js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]"
      }
    }
  }
});
