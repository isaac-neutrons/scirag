"""Upload API routes for document ingestion."""

import logging
from pathlib import Path

from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

from scirag.client.ingest import extract_chunks_from_pdf
from scirag.service.mcp_helpers import call_mcp_tool, run_async

logger = logging.getLogger(__name__)

upload_bp = Blueprint("upload", __name__)

# These will be set by app.py during initialization
local_mcp_server_url = None
upload_folder = None
allowed_extensions = {"pdf"}


def init_upload_routes(mcp_url, folder):
    """Initialize upload routes with dependencies.

    Args:
        mcp_url: Local MCP server URL
        folder: Path to upload folder
    """
    global local_mcp_server_url, upload_folder
    local_mcp_server_url = mcp_url
    upload_folder = folder


def allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed.

    Args:
        filename: The filename to check

    Returns:
        True if extension is allowed, False otherwise
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


@upload_bp.route("/api/collections", methods=["GET"])
def list_collections_endpoint():
    """Get list of existing collection names.

    Returns:
        JSON response with list of collection names
    """
    logger.info("📂 Fetching collection names")
    try:
        collections = run_async(call_mcp_tool(local_mcp_server_url, "list_collections"))

        logger.info(f"✅ Found {len(collections)} collections")
        return jsonify({"success": True, "collections": collections})
    except Exception as e:
        logger.warning(f"⚠️ Could not fetch collections: {e}")
        return jsonify({"success": True, "collections": []})


@upload_bp.route("/api/upload", methods=["POST"])
def upload_documents():
    """Handle document upload and ingestion into vectorstore.

    Expects multipart form data with:
        - files: One or more PDF files
        - collection: Name of the collection to store documents in

    Returns:
        JSON response with upload status
    """
    logger.info("📤 Received document upload request")

    try:
        # Check if files were provided
        if "files" not in request.files:
            logger.warning("❌ No files in request")
            return jsonify({"success": False, "error": "No files provided"}), 400

        files = request.files.getlist("files")
        collection = request.form.get("collection", "default")

        if not files or all(f.filename == "" for f in files):
            logger.warning("❌ No files selected")
            return jsonify({"success": False, "error": "No files selected"}), 400

        logger.info(f"📁 Collection: {collection}")
        logger.info(f"📄 Files received: {len(files)}")

        results = []
        success_count = 0

        for file in files:
            if file.filename == "":
                continue

            if not allowed_file(file.filename):
                results.append(
                    {
                        "filename": file.filename,
                        "status": "error",
                        "error": "File type not allowed. Only PDF files are accepted.",
                    }
                )
                continue

            try:
                # Save file temporarily
                filename = secure_filename(file.filename)
                filepath = upload_folder / filename
                file.save(filepath)
                logger.info(f"💾 Saved file: {filepath}")

                # Extract chunks from PDF (without embeddings)
                chunks = extract_chunks_from_pdf(filepath, collection)
                logger.info(f"📊 Extracted {len(chunks)} chunks from {filename}")

                # Store chunks via MCP tool (handles embedding generation)
                store_result = run_async(
                    call_mcp_tool(
                        local_mcp_server_url,
                        "store_document_chunks",
                        {"chunks": chunks, "collection": collection},
                    )
                )

                if store_result.get("success"):
                    logger.info(
                        f"✅ Stored {store_result.get('chunks_stored', 0)} chunks "
                        f"for {filename} in collection '{collection}'"
                    )
                    results.append(
                        {
                            "filename": filename,
                            "chunks": store_result.get("chunks_stored", len(chunks)),
                            "status": "success",
                        }
                    )
                    success_count += 1
                else:
                    error_msg = store_result.get("message", "Unknown error")
                    logger.error(f"❌ Failed to store chunks: {error_msg}")
                    results.append({"filename": filename, "status": "error", "error": error_msg})

                # Clean up temporary file
                filepath.unlink()

            except Exception as e:
                logger.error(f"❌ Error processing {file.filename}: {e}", exc_info=True)
                results.append({"filename": file.filename, "status": "error", "error": str(e)})

        return jsonify(
            {
                "success": success_count > 0,
                "message": f"Successfully ingested {success_count} of {len(files)} documents",
                "collection": collection,
                "details": results,
            }
        )

    except Exception as e:
        logger.error(f"❌ Error in upload handler: {e}", exc_info=True)
        return jsonify({"success": False, "error": f"Internal server error: {str(e)}"}), 500
