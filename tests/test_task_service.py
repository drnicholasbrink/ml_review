from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ml_review_app import create_app
from ml_review_app.config import TestConfig
from ml_review_app.services.project_service import create_project
from ml_review_app.services.task_service import (
    TaskConflictError,
    TaskManager,
    active_task,
    list_tasks,
    load_task,
)


def make_app(tmp_path: Path, **overrides):
    settings = {"RUNTIME_DIR": tmp_path / "runtime", **overrides}
    config = type("TaskTestConfig", (TestConfig,), settings)
    return create_app(config)


def test_eager_task_tracks_progress_without_persisting_closure_secrets(tmp_path: Path):
    app = make_app(tmp_path)
    manifest = create_project(app.config["RUNTIME_DIR"], "Task review")
    project_path = app.config["RUNTIME_DIR"] / "projects" / manifest["project_id"]
    secret = "never-write-this-api-key"

    def target(progress):
        assert secret
        progress(0, 3, "Preparing")
        progress(2, 3, "Processed two records")
        progress(3, 3, "Saved")

    task = app.extensions["task_manager"].submit(
        project_path,
        kind="fixture",
        title="Fixture task",
        target=target,
        result_url=f"/projects/{manifest['project_id']}",
        failure_message="Fixture failed safely",
    )

    assert task["state"] == "succeeded"
    assert task["percent"] == 100
    assert task["completed"] == 3
    assert active_task(project_path) is None
    persisted = (project_path / "tasks" / f"{task['task_id']}.json").read_text()
    assert secret not in persisted

    client = app.test_client()
    status = client.get(f"/projects/{manifest['project_id']}/tasks/{task['task_id']}.json")
    assert status.status_code == 200
    assert status.headers["Cache-Control"] == "no-store"
    assert status.json["state"] == "succeeded"
    history = client.get(f"/projects/{manifest['project_id']}/tasks")
    assert b"Fixture task" in history.data
    assert b"Succeeded" in history.data


def test_failed_task_exposes_safe_error_and_restart_marks_active_tasks_interrupted(tmp_path: Path):
    app = make_app(tmp_path)
    manifest = create_project(app.config["RUNTIME_DIR"], "Failure review")
    project_path = app.config["RUNTIME_DIR"] / "projects" / manifest["project_id"]

    def target(_progress):
        raise RuntimeError("internal record content must not reach the browser")

    failed = app.extensions["task_manager"].submit(
        project_path,
        kind="fixture",
        title="Failing fixture",
        target=target,
        result_url=f"/projects/{manifest['project_id']}",
        failure_message="Safe failure guidance",
    )
    assert failed["state"] == "failed"
    assert failed["error"] == "Safe failure guidance"
    assert "internal record content" not in json.dumps(failed)

    interrupted = {
        **failed,
        "task_id": "a" * 32,
        "state": "running",
        "error": None,
        "finished_at": None,
    }
    task_path = project_path / "tasks" / f"{interrupted['task_id']}.json"
    task_path.write_text(json.dumps(interrupted))
    TaskManager(app, eager=True)
    recovered = load_task(project_path, interrupted["task_id"])
    assert recovered["state"] == "failed"
    assert recovered["error"] == "Interrupted by application restart"
    assert "Resume the operation" in recovered["message"]


def test_async_task_is_visible_while_running_and_serializes_project_work(tmp_path: Path):
    app = make_app(tmp_path, BACKGROUND_TASKS_EAGER=False)
    manifest = create_project(app.config["RUNTIME_DIR"], "Async review")
    project_path = app.config["RUNTIME_DIR"] / "projects" / manifest["project_id"]
    started = threading.Event()
    release = threading.Event()

    def target(progress):
        progress(1, 4, "Working")
        started.set()
        assert release.wait(timeout=5)
        progress(4, 4, "Done")

    task = app.extensions["task_manager"].submit(
        project_path,
        kind="fixture",
        title="Async fixture",
        target=target,
        result_url=f"/projects/{manifest['project_id']}",
        failure_message="Async fixture failed",
    )
    assert started.wait(timeout=5)
    running = load_task(project_path, task["task_id"])
    assert running["state"] == "running"
    assert running["percent"] == 25
    assert active_task(project_path)["task_id"] == task["task_id"]
    with pytest.raises(TaskConflictError, match="already running"):
        app.extensions["task_manager"].submit(
            project_path,
            kind="second",
            title="Second task",
            target=lambda _progress: None,
            result_url=f"/projects/{manifest['project_id']}",
            failure_message="Second task failed",
        )

    page = app.test_client().get(
        f"/projects/{manifest['project_id']}/tasks?task={task['task_id']}"
    )
    assert b'data-task-result hidden' in page.data
    assert b"Async fixture" in page.data

    blocked = app.test_client().post(f"/projects/{manifest['project_id']}/setup", data={})
    assert blocked.status_code == 409
    assert b"Wait for it to finish" in blocked.data

    release.set()
    for _ in range(100):
        if load_task(project_path, task["task_id"])["state"] == "succeeded":
            break
        threading.Event().wait(0.01)
    assert load_task(project_path, task["task_id"])["state"] == "succeeded"
    assert len(list_tasks(project_path)) == 1
