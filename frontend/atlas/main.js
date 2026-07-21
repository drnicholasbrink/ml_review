import * as duckdb from "@duckdb/duckdb-wasm";
import workerMvp from "@duckdb/duckdb-wasm/dist/duckdb-browser-mvp.worker.js?url";
import wasmMvp from "@duckdb/duckdb-wasm/dist/duckdb-mvp.wasm?url";
import workerEh from "@duckdb/duckdb-wasm/dist/duckdb-browser-eh.worker.js?url";
import wasmEh from "@duckdb/duckdb-wasm/dist/duckdb-eh.wasm?url";
import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
import { EmbeddingAtlas } from "embedding-atlas";

const root = document.getElementById("atlas-root");
const loading = document.getElementById("atlas-loading");
const unsupported = document.getElementById("atlas-unsupported");
const errorPanel = document.getElementById("atlas-error");
const errorMessage = document.getElementById("atlas-error-message");

function browserSupported() {
  if (!("WebAssembly" in window) || !("Worker" in window)) return false;
  const canvas = document.createElement("canvas");
  return Boolean(canvas.getContext("webgl2"));
}

function savedState(key) {
  try {
    return JSON.parse(window.localStorage.getItem(key)) || null;
  } catch (_error) {
    return null;
  }
}

function saveState(key, state) {
  try {
    window.localStorage.setItem(key, JSON.stringify(state));
  } catch (_error) {
    // Exploration state is optional; quota/privacy errors must not break Atlas.
  }
}

function download(bytes, filename, format) {
  const types = {
    csv: "text/csv",
    json: "application/json",
    jsonl: "application/x-ndjson",
    parquet: "application/vnd.apache.parquet"
  };
  const href = URL.createObjectURL(new Blob([bytes], { type: types[format] }));
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(href);
}

async function start() {
  if (!root) return;
  if (!browserSupported()) {
    loading.hidden = true;
    unsupported.hidden = false;
    return;
  }

  let database;
  let connection;
  let component;
  try {
    const bundles = {
      mvp: { mainModule: wasmMvp, mainWorker: workerMvp },
      eh: { mainModule: wasmEh, mainWorker: workerEh }
    };
    const selectedBundle = await duckdb.selectBundle(bundles);
    const candidates = selectedBundle.mainWorker === bundles.mvp.mainWorker
      ? [selectedBundle]
      : [selectedBundle, bundles.mvp];
    const failures = [];
    for (const bundle of candidates) {
      const worker = new Worker(bundle.mainWorker);
      const candidate = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
      let timer;
      try {
        await Promise.race([
          candidate.instantiate(bundle.mainModule, bundle.pthreadWorker),
          new Promise((_, reject) => {
            worker.addEventListener("error", () => reject(new Error("The analytical worker could not start.")), { once: true });
            timer = window.setTimeout(() => reject(new Error("The analytical worker timed out during startup.")), 12000);
          })
        ]);
        window.clearTimeout(timer);
        database = candidate;
        break;
      } catch (error) {
        window.clearTimeout(timer);
        worker.terminate();
        failures.push(error instanceof Error ? error.message : "Unknown worker error");
      }
    }
    if (!database) throw new Error(`Evidence Atlas could not start its local analytical engine. ${failures.join(" ")}`);
    await database.open({ filesystem: { forceFullHTTPReads: true } });
    connection = await database.connect();

    const response = await fetch(root.dataset.url, { credentials: "same-origin" });
    if (!response.ok) throw new Error(`Artifact request failed with status ${response.status}.`);
    await database.registerFileBuffer("evidence-atlas.parquet", new Uint8Array(await response.arrayBuffer()));
    await connection.query("CREATE OR REPLACE TABLE evidence_atlas AS SELECT * FROM 'evidence-atlas.parquet'");
    await database.dropFile("evidence-atlas.parquet");

    const coordinator = new Coordinator();
    const connector = await wasmConnector({ duckdb: database, connection });
    coordinator.databaseConnector(connector);
    const storageKey = root.dataset.storageKey;
    let saveTimer;

    async function exportSelection(predicate, format) {
      const formats = {
        csv: { clause: "(FORMAT CSV)", extension: "csv" },
        json: { clause: "(FORMAT JSON, ARRAY true)", extension: "json" },
        jsonl: { clause: "(FORMAT JSON)", extension: "jsonl" },
        parquet: { clause: "(FORMAT PARQUET)", extension: "parquet" }
      };
      const selected = formats[format];
      if (!selected) throw new Error("Unsupported export format.");
      const filename = `evidence-atlas-selection-${Date.now()}.${selected.extension}`;
      await database.registerEmptyFileBuffer(filename);
      const source = predicate
        ? `(SELECT * FROM evidence_atlas WHERE ${predicate})`
        : "evidence_atlas";
      await coordinator.query(`COPY ${source} TO '${filename}' ${selected.clause}`);
      const bytes = await database.copyFileToBuffer(filename);
      await database.dropFile(filename);
      download(bytes, filename, format);
    }

    component = new EmbeddingAtlas(root, {
      coordinator,
      data: {
        table: root.dataset.table,
        id: "atlas_id",
        projection: { x: "atlas_x", y: "atlas_y" },
        neighbors: "neighbors",
        text: "search_text"
      },
      initialState: savedState(storageKey),
      embeddingViewConfig: { mode: "points", autoLabelEnabled: true },
      defaultChartsConfig: {
        include: ["Year", "Source", "EmbeddingModel", "Cluster", "ai_decision", "ai_confidence"],
        table: true,
        embedding: {}
      },
      onExportSelection: exportSelection,
      onStateChange(state) {
        window.clearTimeout(saveTimer);
        saveTimer = window.setTimeout(() => saveState(storageKey, state), 200);
      }
    });
    loading.hidden = true;
    root.hidden = false;

    window.addEventListener("beforeunload", () => {
      window.clearTimeout(saveTimer);
      component?.destroy();
      connection?.close();
      database?.terminate();
    }, { once: true });
  } catch (error) {
    component?.destroy();
    await connection?.close();
    await database?.terminate();
    loading.hidden = true;
    errorMessage.textContent = error instanceof Error ? error.message : "Reload the page and try again.";
    errorPanel.hidden = false;
    window.dispatchEvent(new CustomEvent("ml-review-atlas-fallback", { detail: { message: errorMessage.textContent } }));
  }
}

start();
