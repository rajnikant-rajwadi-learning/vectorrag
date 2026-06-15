variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Name prefix for all resources."
  type        = string
  default     = "vectorrag"
}

variable "environment" {
  description = "Deployment environment (dev/staging/prod)."
  type        = string
  default     = "dev"
}

variable "image_uri" {
  description = "ECR image URI (with tag/digest) for the Lambda container."
  type        = string
}

variable "openai_api_key" {
  description = "OpenAI API key, stored in Secrets Manager. Pass via TF_VAR_openai_api_key, never commit."
  type        = string
  sensitive   = true
}

variable "chat_model" {
  description = "OpenAI chat model."
  type        = string
  default     = "gpt-4o-mini"
}

variable "embedding_model" {
  description = "OpenAI embedding model."
  type        = string
  default     = "text-embedding-3-small"
}

variable "chroma_s3_uri" {
  description = "Optional s3://bucket/prefix to hydrate the Chroma store from on cold start."
  type        = string
  default     = ""
}

variable "lambda_memory_mb" {
  description = "Lambda memory (MB). More memory = more CPU; helps embedding/cold start."
  type        = number
  default     = 1024
}

variable "lambda_timeout_s" {
  description = "Lambda timeout (seconds)."
  type        = number
  default     = 60
}
