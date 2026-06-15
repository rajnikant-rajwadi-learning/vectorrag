locals {
  name = "${var.project_name}-${var.environment}"
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
  # Parse "s3://bucket/prefix" -> bucket for IAM scoping.
  chroma_bucket = var.chroma_s3_uri != "" ? split("/", replace(var.chroma_s3_uri, "s3://", ""))[0] : ""
}

# --- Secrets Manager: holds the OpenAI API key ---
resource "aws_secretsmanager_secret" "openai" {
  name        = "${local.name}-openai-api-key"
  description = "OpenAI API key for ${local.name}"
  tags        = local.tags
}

resource "aws_secretsmanager_secret_version" "openai" {
  secret_id     = aws_secretsmanager_secret.openai.id
  secret_string = jsonencode({ OPENAI_API_KEY = var.openai_api_key })
}

# --- CloudWatch log group with retention ---
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${local.name}"
  retention_in_days = 30
  tags              = local.tags
}

# --- Lambda function (container image) ---
resource "aws_lambda_function" "api" {
  function_name = local.name
  role          = aws_iam_role.lambda.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_s

  environment {
    variables = {
      VECTORRAG_OPENAI_SECRET_NAME = aws_secretsmanager_secret.openai.name
      VECTORRAG_CHAT_MODEL         = var.chat_model
      VECTORRAG_EMBEDDING_MODEL    = var.embedding_model
      VECTORRAG_CHROMA_DIR         = "/tmp/chroma"
      VECTORRAG_CHROMA_S3_URI      = var.chroma_s3_uri
      # Note: AWS_REGION is provided automatically by the Lambda runtime and is
      # read by the app's config (do not set it here — it is reserved).
      LOG_LEVEL                    = "INFO"
    }
  }

  depends_on = [
    aws_iam_role_policy.lambda_inline,
    aws_cloudwatch_log_group.lambda,
  ]

  tags = local.tags
}

# --- Public HTTPS endpoint via Lambda Function URL ---
# Simpler/cheaper than API Gateway for a single endpoint. Swap to API Gateway
# if you need WAF, custom domains, usage plans, or request throttling per-key.
resource "aws_lambda_function_url" "api" {
  function_name      = aws_lambda_function.api.function_name
  authorization_type = "NONE" # consider "AWS_IAM" for private access

  cors {
    allow_origins = ["*"]
    allow_methods = ["POST", "GET"]
    allow_headers = ["content-type"]
    max_age       = 86400
  }
}
