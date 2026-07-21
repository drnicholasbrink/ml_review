"""Flask app factory."""

from __future__ import annotations

from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFError, CSRFProtect

from .config import Config
from .services.project_service import ensure_runtime_dir


def create_app(config_object: type[Config] | None = None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)
    app.config.from_object(config_object or Config)
    CSRFProtect(app)
    ensure_runtime_dir(app.config["RUNTIME_DIR"])

    from .blueprints.main import bp as main_bp
    from .blueprints.projects import bp as projects_bp
    from .blueprints.pubmed import bp as pubmed_bp
    from .blueprints.imports import bp as imports_bp
    from .blueprints.embeddings import bp as embeddings_bp
    from .blueprints.clustering import bp as clustering_bp
    from .blueprints.screening import bp as screening_bp
    from .blueprints.evaluation import bp as evaluation_bp
    from .blueprints.exports import bp as exports_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(pubmed_bp)
    app.register_blueprint(imports_bp)
    app.register_blueprint(embeddings_bp)
    app.register_blueprint(clustering_bp)
    app.register_blueprint(screening_bp)
    app.register_blueprint(evaluation_bp)
    app.register_blueprint(exports_bp)

    @app.context_processor
    def inject_project_workflow():
        project_id = (request.view_args or {}).get("project_id")
        if not project_id:
            return {}
        from .services.project_service import load_manifest, project_dir
        from .services.workflow_service import build_workflow_steps

        try:
            path = project_dir(app.config["RUNTIME_DIR"], project_id)
            manifest = load_manifest(path)
        except (FileNotFoundError, ValueError):
            return {}
        return {"manifest": manifest, "workflow_steps": build_workflow_steps(path, manifest)}

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

    return app
