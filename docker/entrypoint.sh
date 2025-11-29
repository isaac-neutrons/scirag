#!/bin/bash
# Entrypoint script for SciRAG Docker container
# Starts RavenDB, MCP server, and Flask app using supervisord

set -e

echo "🚀 Starting SciRAG Application Stack..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check required environment variables
if [ "$LLM_SERVICE" = "gemini" ] && [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ ERROR: GEMINI_API_KEY environment variable is required when LLM_SERVICE=gemini"
    exit 1
fi

# Configure RavenDB settings
# Using tar.bz2 installation, so we need to create settings.json
# Setup.Mode must be "Unsecured" (not "None") to allow operations without setup wizard
cat > /usr/lib/ravendb/Server/settings.json <<EOF
{
    "ServerUrl": "http://0.0.0.0:8888",
    "ServerUrl.Tcp": "tcp://0.0.0.0:38888",
    "DataDir": "/var/lib/ravendb/data",
    "Logs.Path": "/var/log/ravendb/logs",
    "Setup.Mode": "Unsecured",
    "Security.UnsecuredAccessAllowed": "PublicNetwork",
    "License.Eula.Accepted": true
}
EOF

echo "✅ RavenDB configuration created"

# Create a script to set up the database after RavenDB starts
cat > /tmp/setup-ravendb.sh <<'DBSETUP'
#!/bin/bash
# Wait for RavenDB to be ready
echo "⏳ Waiting for RavenDB to start..."
for i in {1..30}; do
    if curl -s http://localhost:8888/databases 2>/dev/null | grep -q '\['; then
        echo "✅ RavenDB is ready"
        break
    fi
    sleep 1
done

# Create the scirag database if it doesn't exist
echo "📦 Creating 'scirag' database..."
RESULT=$(curl -s -X PUT "http://localhost:8888/admin/databases?name=scirag&replicationFactor=1" \
    -H "Content-Type: application/json" \
    -d '{"DatabaseName": "scirag"}' 2>&1)

# Check if database was created or already exists
if echo "$RESULT" | grep -q "already exists"; then
    echo "✅ Database 'scirag' already exists"
elif echo "$RESULT" | grep -q "scirag"; then
    echo "✅ Database 'scirag' created successfully"
else
    echo "⚠️ Database creation response: $RESULT"
fi
DBSETUP
chmod +x /tmp/setup-ravendb.sh

# Run database setup in background after supervisord starts
(sleep 5 && /tmp/setup-ravendb.sh) &
echo "✅ Environment:"
echo "   - LLM Service: $LLM_SERVICE"
echo "   - Flask Port: $FLASK_PORT"
echo "   - Local MCP Server: $LOCAL_MCP_SERVER_URL"
echo "   - RavenDB URL: $RAVENDB_URL"
echo "   - RavenDB Data: /var/lib/ravendb/data"
echo "   - Log Level: $LOG_LEVEL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start all services with supervisord
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
