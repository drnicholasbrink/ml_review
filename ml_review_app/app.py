"""Flask app factory."""

from __future__ import annotations

from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFError, CSRFProtect

from .config import Config
from .services.project_service import ensure_runtime_dir
from .services.task_service import TaskManager


def create_app(config_object: type[Config] | None = None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)
    app.config.from_object(config_object or Config)
    CSRFProtect(app)
    ensure_runtime_dir(app.config["RUNTIME_DIR"])
    app.extensions["task_manager"] = TaskManager(
        app,
        eager=bool(app.config.get("BACKGROUND_TASKS_EAGER")),
    )

    from .blueprints.main import bp as main_bp
    from .blueprints.projects import bp as projects_bp
    from .blueprints.pubmed import bp as pubmed_bp
    from .blueprints.imports import bp as imports_bp
    from .blueprints.embeddings import bp as embeddings_bp
    from .blueprints.atlas import bp as atlas_bp
    from .blueprints.clustering import bp as clustering_bp
    from .blueprints.screening import bp as screening_bp
    from .blueprints.evaluation import bp as evaluation_bp
    from .blueprints.extraction import bp as extraction_bp
    from .blueprints.exports import bp as exports_bp
    from .blueprints.tasks import bp as tasks_bp
    from .blueprints.full_texts import bp as full_texts_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(pubmed_bp)
    app.register_blueprint(imports_bp)
    app.register_blueprint(embeddings_bp)
    app.register_blueprint(atlas_bp)
    app.register_blueprint(clustering_bp)
    app.register_blueprint(screening_bp)
    app.register_blueprint(evaluation_bp)
    app.register_blueprint(extraction_bp)
    app.register_blueprint(exports_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(full_texts_bp)

    @app.before_request
    def block_project_mutations_while_task_runs():
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return None
        project_id = (request.view_args or {}).get("project_id")
        if not project_id:
            return None
        from .services.project_service import project_dir
        from .services.task_service import active_task

        try:
            path = project_dir(app.config["RUNTIME_DIR"], project_id)
        except (FileNotFoundError, ValueError):
            return None
        task = active_task(path)
        if task is None:
            return None
        return render_template(
            "error.html",
            title="Background task running",
            message=(
                f"{task['title']} is currently {task['state']}. Wait for it to finish before changing this project."
            ),
        ), 409

    @app.context_processor
    def inject_project_workflow():
        project_id = (request.view_args or {}).get("project_id")
        if not project_id:
            return {}
        from .services.project_service import load_manifest, project_dir
        from .services.task_service import active_task, load_task
        from .services.workflow_service import build_workflow_steps

        try:
            path = project_dir(app.config["RUNTIME_DIR"], project_id)
            manifest = load_manifest(path)
        except (FileNotFoundError, ValueError):
            return {}
        monitored_task = None
        requested_task_id = request.args.get("task", "")
        if requested_task_id:
            try:
                monitored_task = load_task(path, requested_task_id)
            except (FileNotFoundError, ValueError):
                monitored_task = None
        monitored_task = monitored_task or active_task(path)
        return {
            "manifest": manifest,
            "workflow_steps": build_workflow_steps(path, manifest),
            "monitored_task": monitored_task,
        }

    @app.errorhandler(404)
    @app.errorhandler(FileNotFoundError)
    def not_found(_error):
        return render_template(
            "error.html",
            title="Not found",
            message="The requested project or file could not be found.",
        ), 404

    @app.errorhandler(413)
    def upload_too_large(_error):
        max_bytes = app.config["MAX_CONTENT_LENGTH"]
        limit = f"{max_bytes // (1024 * 1024)} MB" if max_bytes >= 1024 * 1024 else f"{max_bytes} byte"
        return render_template(
            "error.html",
            title="Upload too large",
            message=f"The uploaded file exceeds the {limit} limit.",
        ), 413

    @app.errorhandler(CSRFError)
    def csrf_error(_error):
        return render_template(
            "error.html",
            title="Form expired",
            message="This form is missing or has an expired security token. Reload the page and try again.",
        ), 400

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' blob:; "
            "worker-src 'self' blob:; font-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'",
        )
        return response

    return app
