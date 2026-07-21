"""Process-local background jobs with durable, project-scoped status records."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


TASK_DIRECTORY = "tasks"
ACTIVE_STATES = {"queued", "running"}
TERMINAL_STATES = {"succeeded", "failed"}
ProgressCallback = Callable[[int, int, str], None]
TaskCallable = Callable[[ProgressCallback], None]


class TaskConflictError(ValueError):
    """Raised when a project already has a queued or running task."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_path(project_path: Path, task_id: str) -> Path:
    if len(task_id) != 32 or any(character not in "0123456789abcdef" for character in task_id):
        raise FileNotFoundError("Invalid task ID")
    return project_path / TASK_DIRECTORY / f"{task_id}.json"


def _write_task(project_path: Path, task: dict[str, Any]) -> None:
    directory = project_path / TASK_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{task['task_id']}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def load_task(project_path: Path, task_id: str) -> dict[str, Any]:
    path = _task_path(project_path, task_id)
    if not path.is_file():
        raise FileNotFoundError("Task not found")
    return json.loads(path.read_text())


def list_tasks(project_path: Path, *, limit: int = 20) -> list[dict[str, Any]]:
    directory = project_path / TASK_DIRECTORY
    if not directory.is_dir():
        return []
    tasks: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            tasks.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    tasks.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return tasks[:limit]


def active_task(project_path: Path) -> dict[str, Any] | None:
    return next((task for task in list_tasks(project_path) if task.get("state") in ACTIVE_STATES), None)


class TaskManager:
    """Run one background task at a time and persist observable progress as JSON."""

    def __init__(self, app, *, eager: bool = False):
        self.app = app
        self.eager = eager
        self._lock = threading.RLock()
        self._executor = None if eager else ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-review-task")
        self._mark_interrupted_tasks()

    def _mark_interrupted_tasks(self) -> None:
        projects = Path(self.app.config["RUNTIME_DIR"]) / "projects"
        if not projects.is_dir():
            return
        for project_path in projects.iterdir():
            if not project_path.is_dir():
                continue
            for task in list_tasks(project_path, limit=10_000):
                if task.get("state") not in ACTIVE_STATES:
                    continue
                task.update(
                    state="failed",
                    message="The app restarted before this task finished. Resume the operation to continue safely.",
                    error="Interrupted by application restart",
                    finished_at=_now(),
                )
                _write_task(project_path, task)

    def submit(
        self,
        project_path: Path,
        *,
        kind: str,
        title: str,
        target: TaskCallable,
        result_url: str,
        failure_message: str,
    ) -> dict[str, Any]:
        with self._lock:
            current = active_task(project_path)
            if current is not None:
                raise TaskConflictError(f"{current['title']} is already {current['state']} for this project")
            task = {
                "task_id": uuid.uuid4().hex,
                "project_id": project_path.name,
                "kind": kind,
                "title": title,
                "state": "queued",
                "completed": 0,
                "total": 0,
                "percent": 0,
                "message": "Waiting to start",
                "error": None,
                "result_url": result_url,
                "created_at": _now(),
                "started_at": None,
                "updated_at": _now(),
                "finished_at": None,
            }
            _write_task(project_path, task)
            if self.eager:
                self._run(project_path, task["task_id"], target, failure_message)
            else:
                assert self._executor is not None
                self._executor.submit(self._run, project_path, task["task_id"], target, failure_message)
            return load_task(project_path, task["task_id"])

    def _update(self, project_path: Path, task_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            task = load_task(project_path, task_id)
            task.update(changes)
            task["updated_at"] = _now()
            _write_task(project_path, task)
            return task

    def _run(self, project_path: Path, task_id: str, target: TaskCallable, failure_message: str) -> None:
        self._update(
            project_path,
            task_id,
            state="running",
            started_at=_now(),
            message="Starting",
        )

        def progress(completed: int, total: int, message: str) -> None:
            safe_total = max(0, int(total))
            safe_completed = max(0, min(int(completed), safe_total)) if safe_total else max(0, int(completed))
            percent = round((safe_completed / safe_total) * 100) if safe_total else 0
            self._update(
                project_path,
                task_id,
                completed=safe_completed,
                total=safe_total,
                percent=percent,
                message=str(message)[:500],
            )

        try:
            with self.app.app_context():
                target(progress)
        except Exception as exc:  # Background boundary: persist a safe failure state for the UI.
            self.app.logger.exception("Background task %s failed", task_id)
            self._update(
                project_path,
                task_id,
                state="failed",
                error=failure_message,
                message="Task failed. Review the error and resume when ready.",
                finished_at=_now(),
            )
            return
        task = load_task(project_path, task_id)
        completed = task.get("total") or task.get("completed") or 1
        total = task.get("total") or completed
        self._update(
            project_path,
            task_id,
            state="succeeded",
            completed=completed,
            total=total,
            percent=100,
            message="Completed successfully",
            finished_at=_now(),
        )
