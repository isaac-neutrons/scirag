"""Page routes for serving HTML templates."""

from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    """Serve the chat interface web page.

    Returns:
        HTML page with interactive chat interface
    """
    return render_template("chat.html")


@pages_bp.route("/upload")
def upload_page():
    """Serve the document upload page.

    Returns:
        HTML page with drag-and-drop upload interface
    """
    return render_template("upload.html")
