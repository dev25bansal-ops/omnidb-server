"""Resumable downloader for one public HF Xet blob using signed ranged GETs.

Use only for pinned, publicly accessible artifacts when the hub client transfer
is reset mid-file. The signed URL is resolved fresh on each run; ranges are
written to a .part file and fsync'ed after every successful chunk.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import httpx


def download(repo: str, revision: str, filename: str, dest: Path,
             chunk_mib: int = 1, retries: int = 8) -> None:
    resolve = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    timeout = httpx.Timeout(connect=30, read=180, write=60, pool=60)
    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        head = client.head(resolve)
        head.raise_for_status()
        final_url = str(head.url)
        total = int(head.headers.get("content-length", "0"))
        if not total:
            raise RuntimeError("server did not provide content length")
        chunk = chunk_mib * 1024 * 1024
        offset = part.stat().st_size if part.exists() else 0
        if offset > total:
            raise RuntimeError(f"partial file larger than expected: {offset}>{total}")
        with part.open("ab" if offset else "wb") as f:
            while offset < total:
                end = min(total - 1, offset + chunk - 1)
                err = None
                for attempt in range(retries):
                    try:
                        response = client.get(final_url, headers={"Range": f"bytes={offset}-{end}"})
                        if response.status_code != 206:
                            raise RuntimeError(f"expected 206 for range; got {response.status_code}")
                        expected = end - offset + 1
                        if len(response.content) != expected:
                            raise RuntimeError(f"short range {offset}-{end}: got {len(response.content)}, expected {expected}")
                        f.write(response.content)
                        f.flush()
                        os.fsync(f.fileno())
                        offset += expected
                        print(f"{dest.name}: {offset}/{total} ({offset/total:.1%})", flush=True)
                        err = None
                        break
                    except Exception as exc:  # retry exact same byte range
                        err = exc
                        time.sleep(min(30.0, 1.5 ** attempt + random.random()))
                if err is not None:
                    raise RuntimeError(f"failed range {offset}-{end} after {retries} retries") from err
    part.replace(dest)
    print(f"complete: {dest} ({total} bytes)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--filename", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--chunk-mib", type=int, default=1)
    args = ap.parse_args()
    download(args.repo, args.revision, args.filename, Path(args.dest), args.chunk_mib)

if __name__ == "__main__":
    main()
