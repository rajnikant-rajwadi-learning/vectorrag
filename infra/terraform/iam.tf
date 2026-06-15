data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name               = "${local.name}-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
  tags               = local.tags
}

# Basic execution: write logs to CloudWatch.
resource "aws_iam_role_policy_attachment" "logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Least-privilege inline policy: read the one secret, optionally read the S3 prefix.
data "aws_iam_policy_document" "lambda_inline" {
  statement {
    sid       = "ReadOpenAISecret"
    actions   = ["secretsmanager:GetSecretValue"]
    resources = [aws_secretsmanager_secret.openai.arn]
  }

  dynamic "statement" {
    for_each = var.chroma_s3_uri != "" ? [1] : []
    content {
      sid     = "ReadChromaFromS3"
      actions = ["s3:GetObject", "s3:ListBucket"]
      resources = [
        "arn:aws:s3:::${local.chroma_bucket}",
        "arn:aws:s3:::${local.chroma_bucket}/*",
      ]
    }
  }
}

resource "aws_iam_role_policy" "lambda_inline" {
  name   = "${local.name}-inline"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda_inline.json
}
