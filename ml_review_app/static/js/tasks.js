(() => {
  const panel = document.querySelector("[data-task-status-url]");
  if (!panel) return;
  const statusUrl = panel.dataset.taskStatusUrl;
  const taskId = panel.dataset.taskId;
  const progress = panel.querySelector("progress");
  const state = panel.querySelector("[data-task-state]");
  const message = panel.querySelector("[data-task-message]");
  const counts = panel.querySelector("[data-task-counts]");
  const error = panel.querySelector("[data-task-error]");
  const result = panel.querySelector("[data-task-result]");
  const historyRow = document.querySelector(`[data-task-history-id="${taskId}"]`);

  async function poll() {
    try {
      const response = await fetch(statusUrl, { headers: { Accept: "application/json" }, cache: "no-store" });
      if (!response.ok) throw new Error("Task status could not be loaded");
      const task = await response.json();
      progress.value = task.percent || 0;
      progress.setAttribute("value", task.percent || 0);
      state.textContent = task.state;
      state.className = `status-pill task-${task.state}`;
      message.textContent = task.message;
      counts.textContent = task.total ? `${task.completed} of ${task.total}` : `${task.completed} completed`;
      if (task.error) {
        error.textContent = task.error;
        error.hidden = false;
      }
      if (historyRow) {
        const historyState = historyRow.querySelector("[data-task-history-state]");
        const historyCounts = historyRow.querySelector("[data-task-history-counts]");
        const historyMessage = historyRow.querySelector("[data-task-history-message]");
        const historyError = historyRow.querySelector("[data-task-history-error]");
        const historyProgress = historyRow.querySelector("progress");
        historyState.textContent = task.state;
        historyState.className = `status-pill task-${task.state}`;
        historyCounts.textContent = task.total ? `${task.completed} / ${task.total}` : `${task.completed} / —`;
        historyMessage.textContent = task.message;
        historyProgress.value = task.percent || 0;
        historyProgress.setAttribute("value", task.percent || 0);
        if (task.error) {
          historyError.textContent = task.error;
          historyError.hidden = false;
        }
      }
      if (task.state === "succeeded" || task.state === "failed") {
        result.hidden = false;
        return;
      }
    } catch (_error) {
      message.textContent = "Status connection interrupted; retrying…";
    }
    window.setTimeout(poll, 1500);
  }
  window.setTimeout(poll, 500);
})();
