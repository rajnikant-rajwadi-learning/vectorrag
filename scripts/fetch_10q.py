"""Helper to download a 10-Q filing from SEC EDGAR for local testing.

SEC requires a descriptive User-Agent. Example:
    python scripts/fetch_10q.py --url <edgar-filing-url> --out data/raw/acme_10q.htm

This is a convenience for development only; production ingestion reads whatever
files you place under data/raw/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import Request, urlopen

UA = "vectorrag-dev research contact@example.com"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Direct URL to the 10-Q document.")
    ap.add_argument("--out", required=True, help="Output path.")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    req = Request(args.url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:  # noqa: S310 - dev helper, fixed scheme expected
        out.write_bytes(resp.read())
    print(f"Saved {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
