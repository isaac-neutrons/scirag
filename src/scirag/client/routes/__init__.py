"""Flask route blueprints for the scirag client application."""

from scirag.client.routes.chat import chat_bp
from scirag.client.routes.config import get_config, init_config
from scirag.client.routes.health import health_bp
from scirag.client.routes.pages import pages_bp
from scirag.client.routes.upload import upload_bp

__all__ = [
    "chat_bp",
    "health_bp",
    "pages_bp",
    "upload_bp",
    "init_config",
    "get_config",
]
