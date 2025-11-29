#!/bin/bash
# Quick deployment script for SciRAG on Google Cloud Run

set -e

echo "🚀 SciRAG Cloud Run Deployment Script"
echo "======================================"
echo ""

# Check prerequisites
command -v gcloud >/dev/null 2>&1 || { echo "❌ gcloud CLI is required but not installed. Aborting."; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed. Aborting."; exit 1; }

# Get project ID
if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        echo "❌ No GCP project set. Please run: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
fi

echo "📦 Using GCP Project: $PROJECT_ID"

# Get API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY environment variable is not set."
    echo "   Please export your Gemini API key:"
    echo "   export GEMINI_API_KEY=your-api-key-here"
    exit 1
fi

# Configuration
REGION=${REGION:-us-central1}
SERVICE_NAME=${SERVICE_NAME:-scirag}
REPO_NAME=${REPO_NAME:-scirag-repo}
IMAGE_TAG=${IMAGE_TAG:-latest}
MEMORY=${MEMORY:-4Gi}
CPU=${CPU:-2}

echo ""
echo "🔧 Configuration:"
echo "   Region: $REGION"
echo "   Service: $SERVICE_NAME"
echo "   Memory: $MEMORY"
echo "   CPU: $CPU"
echo ""

# Confirm deployment
read -p "Continue with deployment? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

echo ""
echo "📋 Step 1: Enabling required APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com

echo ""
echo "📦 Step 2: Creating Artifact Registry repository..."
gcloud artifacts repositories create $REPO_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="SciRAG Docker images" 2>/dev/null || echo "Repository already exists, continuing..."

echo ""
echo "🔐 Step 3: Creating secret in Secret Manager..."
echo -n "$GEMINI_API_KEY" | gcloud secrets create GEMINI_API_KEY \
    --data-file=- \
    --replication-policy="automatic" 2>/dev/null || {
        echo "Secret already exists, updating..."
        echo -n "$GEMINI_API_KEY" | gcloud secrets versions add GEMINI_API_KEY --data-file=-
    }

# Grant access to secret
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding GEMINI_API_KEY \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" 2>/dev/null || echo "IAM binding already exists"

echo ""
echo "🐋 Step 4: Building and pushing Docker image..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

IMAGE_URL="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:${IMAGE_TAG}"
docker build -t $IMAGE_URL .
docker push $IMAGE_URL

echo ""
echo "🚀 Step 5: Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_URL \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory $MEMORY \
    --cpu $CPU \
    --timeout 300 \
    --port 8080 \
    --set-env-vars "LLM_SERVICE=gemini,GEMINI_MODEL=gemini-2.0-flash-exp,EMBEDDING_MODEL=text-embedding-004,LOG_LEVEL=INFO" \
    --set-secrets "GEMINI_API_KEY=GEMINI_API_KEY:latest"

echo ""
echo "✅ Deployment complete!"
echo ""

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)")
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "🧪 Test your deployment:"
echo "   Health check: curl $SERVICE_URL/health"
echo "   Web interface: open $SERVICE_URL"
echo ""
echo "📊 View logs:"
echo "   gcloud run services logs read $SERVICE_NAME --region $REGION --limit 100"
