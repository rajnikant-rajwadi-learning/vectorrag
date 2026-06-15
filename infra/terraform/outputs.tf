output "function_name" {
  description = "Deployed Lambda function name."
  value       = aws_lambda_function.api.function_name
}

output "function_url" {
  description = "Public HTTPS endpoint for the API."
  value       = aws_lambda_function_url.api.function_url
}

output "secret_name" {
  description = "Secrets Manager secret holding the OpenAI key."
  value       = aws_secretsmanager_secret.openai.name
}

output "log_group" {
  description = "CloudWatch log group."
  value       = aws_cloudwatch_log_group.lambda.name
}
