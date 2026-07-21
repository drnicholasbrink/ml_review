(() => {
  const root = document.getElementById("atlas-root");
  if (!root) return;

  const loading = document.getElementById("atlas-loading");
  const errorPanel = document.getElementById("atlas-error");
  const errorMessage = document.getElementById("atlas-error-message");
  const element = (tag, options = {}) => Object.assign(document.createElement(tag), options);
  const safeText = value => value == null ? "" : String(value);
  const csvCell = value => `"${safeText(value).replace(/"/g, '""')}"`;
  const decisionColors = { include: "#16a34a", uncertain: "#d97706", exclude: "#dc2626" };
  const palette = ["#2563eb", "#7c3aed", "#0891b2", "#ea580c", "#4f46e5", "#0f766e", "#be123c", "#65a30d", "#9333ea", "#0284c7", "#c2410c", "#475569"];

  function savedState() {
    try {
      return JSON.parse(window.localStorage.getItem(root.dataset.storageKey)) || {};
    } catch (_error) {
      return {};
    }
  }

  function saveState(value) {
    try {
      window.localStorage.setItem(root.dataset.storageKey, JSON.stringify(value));
    } catch (_error) {
      // Storage may be disabled in Safari private browsing; exploration still works.
    }
  }

  function categoryColor(value, index, colorBy) {
    if (colorBy === "ai_decision") return decisionColors[safeText(value)] || "#64748b";
    return palette[index % palette.length];
  }

  async function start() {
    try {
      const response = await fetch(root.dataset.previewUrl, { credentials: "same-origin" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "The saved UMAP projection could not be loaded.");
      const rows = payload.rows || [];
      if (!rows.length) throw new Error("The saved UMAP projection does not contain any records.");
      const byId = new Map(rows.map(row => [safeText(row.atlas_id), row]));
      const initial = savedState();

      const controls = element("div", { className: "atlas-native-controls" });
      const searchLabel = element("label", { textContent: "Search citations" });
      const search = element("input", { type: "search", placeholder: "Title, author, DOI, source…", value: safeText(initial.query) });
      searchLabel.appendChild(search);

      const decisionLabel = element("label", { textContent: "Screening decision" });
      const decision = element("select");
      decision.appendChild(element("option", { value: "", textContent: "All decisions" }));
      [...new Set(rows.map(row => safeText(row.ai_decision)).filter(Boolean))].sort().forEach(value => {
        decision.appendChild(element("option", { value, textContent: value }));
      });
      decision.value = safeText(initial.decision);
      decisionLabel.appendChild(decision);

      const clusterLabel = element("label", { textContent: "Cluster" });
      const cluster = element("select");
      cluster.appendChild(element("option", { value: "", textContent: "All clusters" }));
      [...new Set(rows.map(row => row.Cluster).filter(value => value != null))].sort((a, b) => a - b).forEach(value => {
        cluster.appendChild(element("option", { value: String(value), textContent: `Cluster ${value}` }));
      });
      cluster.value = safeText(initial.cluster);
      clusterLabel.appendChild(cluster);

      const colorLabel = element("label", { textContent: "Color points by" });
      const colorBy = element("select");
      [
        ["ai_decision", "Screening decision"],
        ["Cluster", "Cluster"],
        ["Source", "Source"],
        ["EmbeddingModel", "Embedding model"]
      ].forEach(([value, label]) => colorBy.appendChild(element("option", { value, textContent: label })));
      colorBy.value = ["ai_decision", "Cluster", "Source", "EmbeddingModel"].includes(initial.colorBy) ? initial.colorBy : "ai_decision";
      colorLabel.appendChild(colorBy);

      const exportButton = element("button", { type: "button", className: "secondary", textContent: "Export filtered CSV" });
      controls.append(searchLabel, decisionLabel, clusterLabel, colorLabel, exportButton);

      const summary = element("p", { className: "atlas-native-summary", role: "status" });
      const workspace = element("div", { className: "atlas-native-grid" });
      const plotCard = element("section", { className: "card atlas-native-plot-card" });
      const plotHeading = element("div", { className: "atlas-native-plot-heading" });
      const headingText = element("div");
      headingText.append(
        element("h2", { textContent: "UMAP projection" }),
        element("p", { className: "page-intro", textContent: "Coordinates were computed on the server from the stored embeddings. Drag to pan, scroll or use the controls to zoom, and select a point to inspect its citation." })
      );
      const plotActions = element("div", { className: "atlas-plot-actions" });
      const zoomIn = element("button", { type: "button", className: "secondary", textContent: "Zoom in", ariaLabel: "Zoom into UMAP projection" });
      const zoomOut = element("button", { type: "button", className: "secondary", textContent: "Zoom out", ariaLabel: "Zoom out of UMAP projection" });
      const reset = element("button", { type: "button", className: "secondary", textContent: "Reset view" });
      plotActions.append(zoomIn, zoomOut, reset);
      plotHeading.append(headingText, plotActions);

      const canvasWrap = element("div", { className: "atlas-canvas-wrap" });
      const canvas = element("canvas", { className: "atlas-native-canvas" });
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", "Interactive server-computed UMAP projection of review records");
      canvas.textContent = "The UMAP plot requires HTML Canvas, which is supported by current Safari and Chrome releases.";
      const tooltip = element("div", { className: "atlas-canvas-tooltip", hidden: true });
      canvasWrap.append(canvas, tooltip);
      const legend = element("div", { className: "atlas-native-legend", ariaLabel: "UMAP color legend" });
      plotCard.append(plotHeading, canvasWrap, legend);

      const details = element("aside", { className: "card atlas-native-details" });
      details.append(element("h2", { textContent: "Selected record" }), element("p", { textContent: "Choose a point or table row." }));
      workspace.append(plotCard, details);

      const tableCard = element("section", { className: "card atlas-native-table-card" });
      tableCard.appendChild(element("h2", { textContent: "Filtered records" }));
      const tableNote = element("p", { className: "page-intro" });
      const scroll = element("div", { className: "table-scroll atlas-native-table-scroll" });
      const table = element("table", { className: "data-table atlas-native-table" });
      table.innerHTML = "<thead><tr><th>Record</th><th>Year</th><th>Cluster</th><th>Decision</th><th>Confidence</th></tr></thead><tbody></tbody>";
      scroll.appendChild(table);
      tableCard.append(tableNote, scroll);
      root.replaceChildren(controls, summary, workspace, tableCard);

      const xValues = rows.map(row => Number(row.atlas_x)).filter(Number.isFinite);
      const yValues = rows.map(row => Number(row.atlas_y)).filter(Number.isFinite);
      const extent = {
        minX: Math.min(...xValues), maxX: Math.max(...xValues),
        minY: Math.min(...yValues), maxY: Math.max(...yValues)
      };
      let filtered = rows;
      let selected = null;
      let pointPositions = [];
      let view = { zoom: 1, panX: 0, panY: 0 };
      let drag = null;
      let moved = false;

      function categories() {
        const values = [...new Set(filtered.map(row => safeText(row[colorBy.value]) || "Not available"))].sort();
        return new Map(values.map((value, index) => [value, categoryColor(value, index, colorBy.value)]));
      }

      function canvasSize() {
        const bounds = canvasWrap.getBoundingClientRect();
        return { width: Math.max(320, bounds.width), height: Math.max(360, bounds.height) };
      }

      function basePosition(row, width, height) {
        const padding = 28;
        const xRange = extent.maxX - extent.minX || 1;
        const yRange = extent.maxY - extent.minY || 1;
        const x = padding + ((Number(row.atlas_x) - extent.minX) / xRange) * (width - padding * 2);
        const y = height - padding - ((Number(row.atlas_y) - extent.minY) / yRange) * (height - padding * 2);
        return { x, y };
      }

      function transformedPosition(row, width, height) {
        const base = basePosition(row, width, height);
        return {
          x: (base.x - width / 2) * view.zoom + width / 2 + view.panX,
          y: (base.y - height / 2) * view.zoom + height / 2 + view.panY
        };
      }

      function draw() {
        const { width, height } = canvasSize();
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        const context = canvas.getContext("2d");
        context.setTransform(ratio, 0, 0, ratio, 0, 0);
        context.clearRect(0, 0, width, height);
        context.fillStyle = "#f8fafc";
        context.fillRect(0, 0, width, height);
        context.strokeStyle = "#dbe3ef";
        context.strokeRect(0.5, 0.5, width - 1, height - 1);
        const colorMap = categories();
        pointPositions = filtered.map(row => ({ row, ...transformedPosition(row, width, height) }));
        pointPositions.forEach(point => {
          if (point.x < -10 || point.x > width + 10 || point.y < -10 || point.y > height + 10) return;
          const category = safeText(point.row[colorBy.value]) || "Not available";
          context.beginPath();
          context.arc(point.x, point.y, selected === point.row ? 7 : 5, 0, Math.PI * 2);
          context.fillStyle = colorMap.get(category) || "#64748b";
          context.globalAlpha = selected && selected !== point.row ? 0.42 : 0.82;
          context.fill();
          context.globalAlpha = 1;
          context.lineWidth = selected === point.row ? 3 : 1.5;
          context.strokeStyle = selected === point.row ? "#0f172a" : "#ffffff";
          context.stroke();
        });
        legend.replaceChildren();
        [...colorMap.entries()].slice(0, 12).forEach(([label, color]) => {
          const item = element("span");
          const swatch = element("i");
          swatch.style.backgroundColor = color;
          item.append(swatch, document.createTextNode(label));
          legend.appendChild(item);
        });
        if (colorMap.size > 12) legend.appendChild(element("span", { textContent: `+ ${colorMap.size - 12} more categories` }));
      }

      function nearestPoint(event) {
        const bounds = canvas.getBoundingClientRect();
        const x = event.clientX - bounds.left;
        const y = event.clientY - bounds.top;
        let nearest = null;
        let best = 13 * 13;
        pointPositions.forEach(point => {
          const distance = (point.x - x) ** 2 + (point.y - y) ** 2;
          if (distance < best) { nearest = point; best = distance; }
        });
        return nearest;
      }

      async function showRecord(row) {
        selected = row;
        draw();
        details.replaceChildren(element("p", { textContent: "Loading record…" }));
        const response = await fetch(root.dataset.recordUrl.replace("ATLAS_ID", encodeURIComponent(row.atlas_id)), { credentials: "same-origin" });
        const record = await response.json();
        const heading = element("h2", { textContent: safeText(row.Title) || "Untitled record" });
        const metaValues = [row.Authors, row.Journal, row.Date, row.DOI].map(safeText).filter(Boolean);
        const meta = element("p", { className: "record-meta", textContent: metaValues.join(" · ") });
        const provenance = element("p", { className: "atlas-coordinate", textContent: `UMAP coordinates: ${Number(row.atlas_x).toFixed(3)}, ${Number(row.atlas_y).toFixed(3)} · ${safeText(row.EmbeddingModel) || "Embedding model not recorded"}` });
        const links = element("div", { className: "form-actions" });
        if (safeText(row.DOI)) {
          links.appendChild(element("a", { className: "button secondary", href: `https://doi.org/${encodeURIComponent(safeText(row.DOI))}`, target: "_blank", rel: "noopener", textContent: "Open DOI" }));
        }
        const abstractHeading = element("h3", { textContent: "Abstract" });
        const abstract = element("p", { className: "abstract-text", textContent: response.ok ? (safeText(record.Abstract) || "No abstract was supplied.") : "The abstract could not be loaded." });
        const neighborHeading = element("h3", { textContent: "Nearest neighbors in embedding space" });
        const neighborList = element("ol", { className: "atlas-neighbor-list" });
        const neighborIds = (row.neighbors || {}).ids || [];
        const neighborDistances = (row.neighbors || {}).distances || [];
        neighborIds.slice(0, 8).forEach((id, index) => {
          const neighbor = byId.get(safeText(id));
          if (!neighbor) return;
          const item = element("li");
          const button = element("button", { type: "button", className: "study-result" });
          button.append(
            element("span", { textContent: safeText(neighbor.Title) || safeText(neighbor.atlas_id) }),
            element("small", { textContent: Number.isFinite(Number(neighborDistances[index])) ? `Cosine distance ${Number(neighborDistances[index]).toFixed(3)}` : "" })
          );
          button.addEventListener("click", () => showRecord(neighbor));
          item.appendChild(button);
          neighborList.appendChild(item);
        });
        details.replaceChildren(heading, meta, provenance, links, abstractHeading, abstract, neighborHeading, neighborList);
      }

      function render() {
        const query = search.value.trim().toLowerCase();
        filtered = rows.filter(row => {
          const haystack = [row.Title, row.Authors, row.Journal, row.DOI, row.Source].map(safeText).join(" ").toLowerCase();
          return (!query || haystack.includes(query))
            && (!decision.value || safeText(row.ai_decision) === decision.value)
            && (!cluster.value || safeText(row.Cluster) === cluster.value);
        });
        summary.textContent = `${filtered.length} of ${rows.length} records · browser-native Canvas 2D · server-computed UMAP`;
        tableNote.textContent = filtered.length > 250 ? `Showing the first 250 of ${filtered.length} matching records in the table; all matching points remain visible in the plot and export.` : `${filtered.length} matching records.`;
        const body = table.querySelector("tbody");
        body.replaceChildren();
        filtered.slice(0, 250).forEach(row => {
          const tr = element("tr");
          tr.tabIndex = 0;
          [row.Title || row.atlas_id, row.Year, row.Cluster, row.ai_decision, row.ai_confidence].forEach(value => tr.appendChild(element("td", { textContent: safeText(value) })));
          tr.addEventListener("click", () => showRecord(row));
          tr.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") showRecord(row); });
          body.appendChild(tr);
        });
        saveState({ query: search.value, decision: decision.value, cluster: cluster.value, colorBy: colorBy.value });
        draw();
      }

      function zoom(factor) {
        view.zoom = Math.max(0.5, Math.min(12, view.zoom * factor));
        draw();
      }
      zoomIn.addEventListener("click", () => zoom(1.35));
      zoomOut.addEventListener("click", () => zoom(1 / 1.35));
      reset.addEventListener("click", () => { view = { zoom: 1, panX: 0, panY: 0 }; draw(); });
      canvas.addEventListener("wheel", event => {
        event.preventDefault();
        zoom(event.deltaY < 0 ? 1.15 : 1 / 1.15);
      }, { passive: false });
      canvas.addEventListener("pointerdown", event => {
        drag = { x: event.clientX, y: event.clientY, panX: view.panX, panY: view.panY };
        moved = false;
        if (canvas.setPointerCapture) canvas.setPointerCapture(event.pointerId);
      });
      canvas.addEventListener("pointermove", event => {
        if (drag) {
          const dx = event.clientX - drag.x;
          const dy = event.clientY - drag.y;
          moved = moved || Math.abs(dx) + Math.abs(dy) > 4;
          view.panX = drag.panX + dx;
          view.panY = drag.panY + dy;
          draw();
          return;
        }
        const point = nearestPoint(event);
        if (!point) { tooltip.hidden = true; return; }
        const bounds = canvasWrap.getBoundingClientRect();
        tooltip.textContent = safeText(point.row.Title) || safeText(point.row.atlas_id);
        tooltip.style.left = `${Math.max(8, Math.min(event.clientX - bounds.left + 12, bounds.width - 260))}px`;
        tooltip.style.top = `${Math.max(8, event.clientY - bounds.top - 36)}px`;
        tooltip.hidden = false;
      });
      canvas.addEventListener("pointerleave", () => { tooltip.hidden = true; });
      canvas.addEventListener("pointerup", event => {
        if (!moved) {
          const point = nearestPoint(event);
          if (point) showRecord(point.row);
        }
        drag = null;
      });
      canvas.addEventListener("pointercancel", () => {
        drag = null;
      });

      [search, decision, cluster, colorBy].forEach(control => control.addEventListener(control === search ? "input" : "change", render));
      exportButton.addEventListener("click", () => {
        const columns = ["atlas_id", "Title", "Authors", "Journal", "Date", "DOI", "Source", "EmbeddingModel", "Year", "Cluster", "ai_decision", "ai_confidence", "ai_exclusion_reason", "atlas_x", "atlas_y"];
        const csv = [columns.map(csvCell).join(","), ...filtered.map(row => columns.map(column => csvCell(row[column])).join(","))].join("\n");
        const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
        const anchor = element("a", { href: url, download: "evidence-atlas-filtered.csv" });
        anchor.click();
        window.setTimeout(() => URL.revokeObjectURL(url), 0);
      });

      if ("ResizeObserver" in window) new ResizeObserver(draw).observe(canvasWrap);
      else window.addEventListener("resize", draw);
      render();
      loading.hidden = true;
      root.hidden = false;
    } catch (error) {
      loading.hidden = true;
      errorMessage.textContent = error instanceof Error ? error.message : "Reload the page and try again.";
      errorPanel.hidden = false;
    }
  }

  start();
})();
