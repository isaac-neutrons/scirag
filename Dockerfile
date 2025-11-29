# Multi-stage Dockerfile for SciRAG application with RavenDB
# Designed for deployment on Google Cloud Run

# Stage 1: Python application builder
FROM python:3.12-slim AS python-builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Stage 2: Final runtime image
# Using .NET runtime-deps base image for RavenDB compatibility
# Runtime stage - use Python 3.12 slim to match builder
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime dependencies for RavenDB and Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssl \
    jq \
    curl \
    wget \
    supervisor \
    bzip2 \
    libicu-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install RavenDB tar.bz2 (official releases)
# Note: Using RavenDB 7.x for vector search support (required by ravendb Python client >=7.1.0)
ENV RAVENDB_VERSION=7.0.1
RUN wget -q https://daily-builds.s3.amazonaws.com/RavenDB-${RAVENDB_VERSION}-linux-x64.tar.bz2 && \
    tar xjf RavenDB-${RAVENDB_VERSION}-linux-x64.tar.bz2 && \
    mv RavenDB /usr/lib/ravendb && \
    rm RavenDB-${RAVENDB_VERSION}-linux-x64.tar.bz2

# Create RavenDB directories (mimicking .deb package structure)
RUN mkdir -p /var/lib/ravendb/data /var/log/ravendb/logs /etc/ravendb

# Copy Python packages from builder (same Python 3.12 version)
COPY --from=python-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Create directories for logs and documents
RUN mkdir -p /app/logs /app/documents /var/log/supervisor

# RavenDB directories are created by the .deb package
# Default paths from official RavenDB Dockerfile:
# - /var/lib/ravendb/data (data)
# - /var/lib/ravendb/nuget (nuget packages)
# - /var/log/ravendb/logs (logs)
# - /etc/ravendb (config)

# Copy supervisor configuration
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Copy entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Environment variables with defaults
# RavenDB official environment variables
ENV RAVEN_ARGS='' \
    RAVEN_SETTINGS='' \
    RAVEN_IN_DOCKER='true' \
    RAVEN_Setup_Mode='Initial' \
    RAVEN_ServerUrl_Tcp='38888' \
    RAVEN_AUTO_INSTALL_CA='true' \
    RAVEN_DataDir='/var/lib/ravendb/data' \
    RAVEN_Logs_Path='/var/log/ravendb/logs' \
    RAVEN_Security_UnsecuredAccessAllowed='PublicNetwork' \
    RAVEN_ServerUrl='http://0.0.0.0:8888'

# SciRAG application environment variables
ENV RAVENDB_URL=http://localhost:8888 \
    RAVENDB_DATABASE=scirag \
    LOCAL_MCP_SERVER_URL=http://localhost:8001/sse \
    LLM_SERVICE=gemini \
    LLM_MODEL=gemini-2.5-flash \
    EMBEDDING_MODEL=gemini-embedding-001 \
    EMBEDDING_DIMENSIONS=3072 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=8080 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

# Expose ports
# 8080: Flask app (Cloud Run expects this port by default)
# 8001: MCP server
# 8888: RavenDB HTTP
# 38888: RavenDB TCP
EXPOSE 8080 8001 8888 38888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Use entrypoint script to start all services
ENTRYPOINT ["/app/entrypoint.sh"]
