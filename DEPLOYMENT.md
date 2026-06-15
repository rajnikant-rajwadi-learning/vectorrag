# Deployment Guide — AWS Lambda

This deploys the VectorRAG API as a **container-image Lambda** fronted by a
**Lambda Function URL**, with the OpenAI key in **Secrets Manager**, provisioned by
**Terraform** and shipped by a **CI/CD pipeline**.

```
GitHub push ─▶ CI (test/lint) ─▶ build image ─▶ ECR ─▶ Terraform apply ─▶ Lambda
                                                                   │
                       Secrets Manager (OpenAI key)  ◀─────────────┤
                       S3 (Chroma store)  ──hydrate /tmp──▶ Lambda ┘
```

Why a **container image** (not a zip layer): `chromadb` plus native deps exceed the
250 MB unzipped layer limit. Container images allow up to 10 GB. The image is built
with **uv** (`uv export` from `uv.lock` → installed into the Lambda task root), so
the deployed dependency set is identical to what the lockfile pins.

---

## 1. Prerequisites

- AWS account + AWS CLI configured (`aws sts get-caller-identity` works)
- Docker
- Terraform ≥ 1.6
- [uv](https://docs.astral.sh/uv/) (for local ingest / tests; the image build uses uv internally)
- An OpenAI API key

```bash
export AWS_REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

---

## 2. Build & push the image to ECR

```bash
# Create the repo (first time only)
aws ecr create-repository --repository-name vectorrag \
  --image-scanning-configuration scanOnPush=true --region $AWS_REGION

# Authenticate Docker to ECR
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $ECR

# Build (must be linux/amd64 for Lambda) and push
docker build --platform linux/amd64 -t $ECR/vectorrag:latest .
docker push $ECR/vectorrag:latest
```

---

## 3. (Optional) Publish the vector store to S3

Lambda's filesystem is read-only except `/tmp`. The function hydrates Chroma from
S3 on cold start. Ingest locally first, then sync:

```bash
uv run vectorrag ingest data/raw          # builds ./.chroma locally
aws s3 sync ./.chroma s3://my-bucket/vectorrag/chroma
```

Then set `chroma_s3_uri = "s3://my-bucket/vectorrag/chroma"` in Terraform. If you
skip this, deploy with an empty store and ingest via another path (e.g. an EFS
mount or a separate ingestion Lambda/job).

> Cold-start tip: keep the store small, raise `lambda_memory_mb` (more CPU), and
> consider Provisioned Concurrency for latency-sensitive use.

---

## 4. Provision with Terraform

```bash
cd infra/terraform
terraform init

# Provide the OpenAI key via env var (don't put it in a committed file):
export TF_VAR_openai_api_key="sk-..."

terraform apply \
  -var="image_uri=$ECR/vectorrag:latest" \
  -var="aws_region=$AWS_REGION" \
  -var="environment=prod"
  # -var="chroma_s3_uri=s3://my-bucket/vectorrag/chroma"   # if using S3
```

Terraform creates:
- `aws_secretsmanager_secret` — holds the OpenAI key (JSON `{OPENAI_API_KEY}`)
- `aws_lambda_function` (container image) with least-privilege IAM
- `aws_lambda_function_url` — public HTTPS endpoint
- `aws_cloudwatch_log_group` — 30-day retention

Get the endpoint:

```bash
terraform output -raw function_url
```

Test it:

```bash
URL=$(terraform output -raw function_url)
curl -s "${URL}ask" -H "content-type: application/json" \
  -d '{"question":"What was net income for the quarter?"}' | jq
```

To tear down: `terraform destroy` (same `-var`s).

---

## 5. CI/CD pipeline

Two equivalent options are included — pick one.

### Option A — GitHub Actions (`.github/workflows/`)

- **`ci.yml`** runs on every push/PR: ruff, mypy, pytest.
- **`deploy.yml`** runs on `main`: builds & pushes the image to ECR, then
  `terraform apply`. Uses **GitHub OIDC** to assume an AWS role — no static AWS
  keys stored in GitHub.

Set up once:

1. Create an IAM role trusting GitHub's OIDC provider
   (`token.actions.githubusercontent.com`) with permissions for ECR, Lambda,
   IAM (pass-role), Secrets Manager, CloudWatch Logs, and S3 (if used).
2. Add repo **secrets**:
   - `AWS_DEPLOY_ROLE_ARN` — the role from step 1
   - `OPENAI_API_KEY` — your OpenAI key

Push to `main` → it tests, builds, and deploys.

### Option B — AWS CodePipeline + CodeBuild (`infra/pipeline/buildspec.yml`)

For AWS-native shops. Create a CodePipeline with a source stage (GitHub/CodeCommit)
and a CodeBuild stage using `infra/pipeline/buildspec.yml`. Grant the CodeBuild
role the same permissions as above, and inject `TF_VAR_openai_api_key` from Secrets
Manager via the `env.secrets-manager` block (already stubbed in the buildspec).

---

## 6. Production hardening checklist

- [ ] **Auth on the endpoint** — switch the Function URL to `AWS_IAM`, or front it
      with **API Gateway + WAF** and per-key usage plans/throttling.
- [ ] **Rate limiting** — API Gateway throttling or a WAF rate rule to cap cost.
- [ ] **Tighten CORS** — replace `allow_origins=["*"]` with your domain(s).
- [ ] **Secret rotation** — enable rotation on the Secrets Manager secret.
- [ ] **Remote TF state** — enable the S3 backend block in `versions.tf` with
      DynamoDB locking.
- [ ] **Alarms** — CloudWatch alarms on Lambda errors, throttles, p99 duration.
- [ ] **Provisioned Concurrency** — if cold starts matter.
- [ ] **Cost guardrails** — AWS Budgets + monitor OpenAI usage; output tokens are
      already capped in app config.
