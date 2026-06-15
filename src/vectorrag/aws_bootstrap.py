"""AWS cold-start helpers.

Hydrates the Chroma persistence directory from an S3 prefix into Lambda's writable
``/tmp`` so the function can serve queries without bundling the DB in the image.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from .config import get_settings
from .logging_config import get_logger

log = get_logger(__name__)


def hydrate_chroma_from_s3() -> None:
    """Download s3://bucket/prefix/* into VECTORRAG_CHROMA_DIR if not present.

    Controlled by env var ``VECTORRAG_CHROMA_S3_URI``. Safe to call once per cold
    start; it no-ops if the local directory already has data (warm container).
    """
    s3_uri = os.getenv("VECTORRAG_CHROMA_S3_URI")
    if not s3_uri:
        return

    settings = get_settings()
    dest = Path(settings.chroma_dir)
    if dest.exists() and any(dest.iterdir()):
        log.info("chroma_already_hydrated", dir=str(dest))
        return

    import boto3

    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    dest.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=settings.aws_region)
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(target))
            count += 1
    log.info("chroma_hydrated", files=count, dir=str(dest))
