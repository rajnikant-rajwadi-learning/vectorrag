# Copy to a private *.tfvars (gitignored) and fill in, OR pass via TF_VAR_* env vars.
# Do NOT commit real secrets.

aws_region      = "us-east-1"
project_name    = "vectorrag"
environment     = "dev"

# Set after building & pushing the image (see DEPLOYMENT.md):
image_uri       = "<account-id>.dkr.ecr.us-east-1.amazonaws.com/vectorrag:latest"

# Prefer: export TF_VAR_openai_api_key=sk-...   (keep it out of files)
# openai_api_key = "sk-..."

chat_model      = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"

# Optional: hydrate the vector store from S3 at cold start.
# chroma_s3_uri = "s3://my-bucket/vectorrag/chroma"

lambda_memory_mb = 1024
lambda_timeout_s = 60
