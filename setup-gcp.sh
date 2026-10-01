# 1. Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# 2. Set your Active GCP Project ID (replace YOUR_PROJECT_ID)
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"
gcloud config set project $PROJECT_ID

# 3. Enable Required GCP APIs
gcloud services enable \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com

# 4. Create a Cloud Storage Bucket for Code & Artifacts
export BUCKET_NAME="${PROJECT_ID}-pytorch-ddp"
gcloud storage buckets create gs://${BUCKET_NAME} --location=${REGION}