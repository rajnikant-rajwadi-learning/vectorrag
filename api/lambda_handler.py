"""AWS Lambda entry point.

Wraps the FastAPI app with Mangum so API Gateway / Lambda Function URL events are
translated to ASGI. Set the Lambda handler to ``api.lambda_handler.handler``.

Cold-start note: the Chroma collection should be available on the local
filesystem. For Lambda, hydrate ``VECTORRAG_CHROMA_DIR`` (e.g. /tmp/chroma) from
S3 at init, or mount EFS. See DEPLOYMENT.md.
"""

from __future__ import annotations

import os

from mangum import Mangum

from .app import app

# Optionally hydrate the vector store from S3 on cold start.
if os.getenv("VECTORRAG_CHROMA_S3_URI"):
    from vectorrag.aws_bootstrap import hydrate_chroma_from_s3

    hydrate_chroma_from_s3()

handler = Mangum(app, lifespan="off")
