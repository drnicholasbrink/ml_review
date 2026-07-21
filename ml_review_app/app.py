"""Flask app factory."""

from __future__ import annotations

from flask import Flask

from .config import Config
from .services.project_service import ensure_runtime_dir


def create_app(config_object: type[Config] | None = None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)
    app.config.from_object(config_object or Config)
    ensure_runtime_dir(app.config["RUNTIME_DIR"])

    from .blueprints.main import bp as main_bp
    from .blueprints.projects import bp as projects_bp
    from .blueprints.pubmed import bp as pubmed_bp
    from .blueprints.imports import bp as imports_bp
    from .blueprints.embeddings import bp as embeddings_bp
    from .blueprints.clustering import bp as clustering_bp
    from .blueprints.screening import bp as screening_bp
    from .blueprints.exports import bp as exports_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(projects_bp)
    app.register_blueprint(pubmed_bp)
    app.register_blueprint(imports_bp)
    app.register_blueprint(embeddings_bp)
    app.register_blueprint(clustering_bp)
    app.register_blueprint(screening_bp)
    app.register_blueprint(exports_bp)

    return app
