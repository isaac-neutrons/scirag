"""Flask route blueprints for the scirag client application."""

from scirag.client.routes.chat import chat_bp, init_chat_routes
from scirag.client.routes.health import health_bp, init_health_routes
from scirag.client.routes.pages import pages_bp
from scirag.client.routes.upload import init_upload_routes, upload_bp

__all__ = [
    "chat_bp",
    "health_bp",
    "pages_bp",
    "upload_bp",
    "init_chat_routes",
    "init_health_routes",
    "init_upload_routes",
]
