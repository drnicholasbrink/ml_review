(() => {
  const container = document.getElementById("atlas-fallback");
  if (!container) return;
  let activated = false;

  const element = (tag, options = {}) => Object.assign(document.createElement(tag), options);
  const safeText = value => value == null ? "" : String(value);
  const csvCell = value => `"${safeText(value).replaceAll('"', '""')}"`;

  async function activate(event) {
    if (activated) return;
    activated = true;
    const loading = document.getElementById("atlas-loading");
    const error = document.getElementById("atlas-error");
    if (loading) loading.hidden = true;
    if (error) error.hidden = true;
    container.hidden = false;

    const notice = element("p", {
      className: "notice",
      textContent: "The high-performance Atlas engine is unavailable in this browser. This compatible view uses the same saved projection and supports citation search, filters, record inspection, nearest-neighbor navigation, and CSV export."
    });
    const status = element("p", { className: "page-intro", textContent: "Loading compatible Atlas…" });
    container.append(notice, status);

    try {
      const response = await fetch(container.dataset.previewUrl, { credentials: "same-origin" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Compatible Atlas data could not be loaded.");
      const rows = payload.rows || [];
      const byId = new Map(rows.map(row => [safeText(row.atlas_id), row]));
      const controls = element("div", { className: "atlas-compatible-controls" });
      const searchLabel = element("label", { textContent: "Search citations" });
      const search = element("input", { type: "search", placeholder: "Title, author, DOI, source…" });
      searchLabel.appendChild(search);
      const decisionLabel = element("label", { textContent: "Screening decision" });
      const decision = element("select");
      decision.appendChild(element("option", { value: "", textContent: "All decisions" }));
      [...new Set(rows.map(row => safeText(row.ai_decision)).filter(Boolean))].sort().forEach(value => decision.appendChild(element("option", { value, textContent: value })));
      decisionLabel.appendChild(decision);
      const clusterLabel = element("label", { textContent: "Cluster" });
      const cluster = element("select");
      cluster.appendChild(element("option", { value: "", textContent: "All clusters" }));
      [...new Set(rows.map(row => row.Cluster).filter(value => value != null))].sort((a, b) => a - b).forEach(value => cluster.appendChild(element("option", { value: String(value), textContent: `Cluster ${value}` })));
      clusterLabel.appendChild(cluster);
      const exportButton = element("button", { type: "button", className: "secondary", textContent: "Export filtered CSV" });
      controls.append(searchLabel, decisionLabel, clusterLabel, exportButton);

      const summary = element("p", { className: "page-intro", role: "status" });
      const workspace = element("div", { className: "atlas-compatible-grid" });
      const plotCard = element("section", { className: "card" });
      plotCard.appendChild(element("h2", { textContent: "Saved UMAP projection" }));
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 800 480");
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", "Interactive UMAP projection of review records");
      svg.classList.add("atlas-compatible-plot");
      plotCard.appendChild(svg);
      const details = element("aside", { className: "card atlas-compatible-details" });
      details.append(element("h2", { textContent: "Selected record" }), element("p", { textContent: "Choose a point or table row." }));
      workspace.append(plotCard, details);

      const tableCard = element("section", { className: "card" });
      tableCard.appendChild(element("h2", { textContent: "Filtered records" }));
      const scroll = element("div", { className: "table-scroll" });
      const table = element("table", { className: "data-table atlas-compatible-table" });
      table.innerHTML = "<thead><tr><th>Record</th><th>Year</th><th>Cluster</th><th>Decision</th><th>Confidence</th></tr></thead><tbody></tbody>";
      scroll.appendChild(table); tableCard.appendChild(scroll);
      container.replaceChildren(notice, controls, summary, workspace, tableCard);

      const colors = { include: "#16a34a", uncertain: "#d97706", exclude: "#dc2626" };
      const extent = key => {
        const values = rows.map(row => Number(row[key])).filter(Number.isFinite);
        return [Math.min(...values), Math.max(...values)];
      };
      const [minX, maxX] = extent("atlas_x");
      const [minY, maxY] = extent("atlas_y");
      const scale = (value, min, max, start, end) => min === max ? (start + end) / 2 : start + (Number(value) - min) * (end - start) / (max - min);
      let filtered = rows;

      async function showRecord(row) {
        details.replaceChildren(element("p", { textContent: "Loading record…" }));
        const response = await fetch(container.dataset.recordUrl.replace("ATLAS_ID", encodeURIComponent(row.atlas_id)), { credentials: "same-origin" });
        const record = await response.json();
        const heading = element("h2", { textContent: safeText(row.Title) || "Untitled record" });
        const meta = element("p", { className: "record-meta", textContent: [row.Authors, row.Journal, row.Date, row.DOI].map(safeText).filter(Boolean).join(" · ") });
        const links = element("div", { className: "form-actions" });
        if (safeText(row.DOI)) {
          const doi = element("a", { className: "button secondary", href: `https://doi.org/${encodeURIComponent(safeText(row.DOI))}`, target: "_blank", rel: "noopener", textContent: "Open DOI" });
          links.appendChild(doi);
        }
        const abstractHeading = element("h3", { textContent: "Abstract" });
        const abstract = element("p", { className: "abstract-text", textContent: response.ok ? (safeText(record.Abstract) || "No abstract was supplied.") : "The abstract could not be loaded." });
        const neighborHeading = element("h3", { textContent: "Nearest neighbors" });
        const neighborList = element("ol", { className: "atlas-neighbor-list" });
        ((row.neighbors || {}).ids || []).slice(0, 5).forEach(id => {
          const neighbor = byId.get(safeText(id));
          if (!neighbor) return;
          const item = element("li");
          const button = element("button", { type: "button", className: "study-result", textContent: safeText(neighbor.Title) || safeText(neighbor.atlas_id) });
          button.addEventListener("click", () => showRecord(neighbor)); item.appendChild(button); neighborList.appendChild(item);
        });
        details.replaceChildren(heading, meta, links, abstractHeading, abstract, neighborHeading, neighborList);
      }

      function render() {
        const query = search.value.trim().toLowerCase();
        filtered = rows.filter(row => {
          const haystack = [row.Title, row.Authors, row.Journal, row.DOI, row.Source].map(safeText).join(" ").toLowerCase();
          return (!query || haystack.includes(query)) && (!decision.value || safeText(row.ai_decision) === decision.value) && (!cluster.value || safeText(row.Cluster) === cluster.value);
        });
        summary.textContent = `${filtered.length} of ${rows.length} records shown`;
        svg.replaceChildren();
        filtered.forEach(row => {
          const point = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          point.setAttribute("cx", scale(row.atlas_x, minX, maxX, 32, 768));
          point.setAttribute("cy", scale(row.atlas_y, minY, maxY, 448, 32));
          point.setAttribute("r", "7");
          point.setAttribute("fill", colors[safeText(row.ai_decision)] || "#2563eb");
          point.setAttribute("tabindex", "0"); point.setAttribute("role", "button"); point.setAttribute("aria-label", safeText(row.Title) || safeText(row.atlas_id));
          point.addEventListener("click", () => showRecord(row));
          point.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") showRecord(row); });
          svg.appendChild(point);
        });
        const body = table.querySelector("tbody"); body.replaceChildren();
        filtered.slice(0, 250).forEach(row => {
          const tr = element("tr"); tr.tabIndex = 0;
          [row.Title || row.atlas_id, row.Year, row.Cluster, row.ai_decision, row.ai_confidence].forEach(value => tr.appendChild(element("td", { textContent: safeText(value) })));
          tr.addEventListener("click", () => showRecord(row));
          tr.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") showRecord(row); });
          body.appendChild(tr);
        });
      }

      [search, decision, cluster].forEach(control => control.addEventListener(control === search ? "input" : "change", render));
      exportButton.addEventListener("click", () => {
        const columns = ["atlas_id", "Title", "Authors", "Journal", "Date", "DOI", "Source", "Year", "Cluster", "ai_decision", "ai_confidence", "ai_exclusion_reason"];
        const csv = [columns.map(csvCell).join(","), ...filtered.map(row => columns.map(column => csvCell(row[column])).join(","))].join("\n");
        const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
        const anchor = element("a", { href: url, download: "evidence-atlas-filtered.csv" }); anchor.click(); URL.revokeObjectURL(url);
      });
      render();
    } catch (error) {
      status.className = "notice";
      status.textContent = error instanceof Error ? error.message : "Compatible Atlas could not be loaded.";
    }
  }

  window.addEventListener("ml-review-atlas-fallback", activate, { once: true });
})();
