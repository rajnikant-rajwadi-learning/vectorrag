# AWS Lambda container image for the VectorRAG API, built with uv.
# Built as a container because chromadb + native deps exceed the 250MB zipped
# layer limit; container images allow up to 10GB.

FROM public.ecr.aws/lambda/python:3.12

# Bring in the uv binary (pinned tag for reproducible builds).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

# Resolve dependencies from the lockfile (reproducible) and install them straight
# into the Lambda task root. Copy lock + project metadata first for layer caching.
COPY pyproject.toml uv.lock ${LAMBDA_TASK_ROOT}/
RUN cd ${LAMBDA_TASK_ROOT} \
    && uv export --frozen --no-dev --no-emit-project --format requirements-txt -o /tmp/requirements.txt \
    && uv pip install --system --target "${LAMBDA_TASK_ROOT}" -r /tmp/requirements.txt

# Copy application code.
COPY src/ ${LAMBDA_TASK_ROOT}/src/
COPY api/ ${LAMBDA_TASK_ROOT}/api/

# Make the src layout importable.
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}/src

# Lambda handler: module.path.callable
CMD ["api.lambda_handler.handler"]
