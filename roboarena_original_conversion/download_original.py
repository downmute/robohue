"""Download only the immutable selection, using the public Git LFS batch API."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from original_io import (
    positive_int,
    read_json,
    selected_files,
    source_path,
    verify_source,
    write_json,
)

BASE = Path(__file__).resolve().parent
gate = threading.Lock()
cooldown_until = 0.0


def request(url, body=None, headers=None):
    global cooldown_until
    for attempt in range(7):
        with gate:
            delay = max(0, cooldown_until - time.monotonic())
        if delay:
            time.sleep(delay)
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=headers or {}),
                timeout=90,
            )
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 6:
                raise RuntimeError(
                    f"Public source request returned HTTP {e.code}"
                ) from None
            delay = max(2**attempt, int(e.headers.get("Retry-After", "10")))
            with gate:
                cooldown_until = max(cooldown_until, time.monotonic() + delay)
        except (OSError, TimeoutError):
            if attempt == 6:
                raise RuntimeError(
                    "Public source transfer failed after retries"
                ) from None
            time.sleep(min(30, 2**attempt))


def transfer(root, entry, action):
    """Verify a temporary download before replacing the destination file."""
    path = source_path(root, entry["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    try:
        with (
            request(action["href"], headers=action.get("header")) as response,
            temp.open("wb") as out,
        ):
            h = hashlib.sha256()
            size = 0
            while block := response.read(1024 * 1024):
                out.write(block)
                h.update(block)
                size += len(block)
        if size != entry["bytes"] or h.hexdigest() != entry["sha256"]:
            raise ValueError("Downloaded bytes differ from pinned source digest")
        os.replace(temp, path)
        return {"path": entry["path"], "status": "downloaded"}
    except Exception as e:  # noqa: BLE001 - Clean up partial files and report each transfer failure.
        temp.unlink(missing_ok=True)
        return {
            "path": entry["path"],
            "status": "failed",
            "error": type(e).__name__ + ": " + str(e),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="convert/download the first N selected episodes",
    )
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=12,
        help="parallel downloads (default: 12)",
    )
    args = parser.parse_args()
    manifest = read_json(BASE / "download_manifest.json")
    records = read_json(BASE / "selection.json")["records"]
    if args.limit is not None:
        records = records[: args.limit]
    entries = selected_files(manifest, records)
    root = BASE / "assets/roboarena"
    results, pending = [], []
    for entry in entries:
        try:
            verify_source(root, entry)
            results.append({"path": entry["path"], "status": "already_verified"})
        except ValueError:
            pending.append(entry)
    print(
        json.dumps(
            {
                "selected_episodes": len(records),
                "files": len(entries),
                "pending": len(pending),
                "pending_bytes": sum(e["bytes"] for e in pending),
            }
        ),
        flush=True,
    )
    # Metadata uses Git blobs rather than LFS, and is normally already cached.
    lfs = [e for e in pending if e["kind"] != "metadata"]
    for e in pending:
        if e["kind"] == "metadata":
            results.append(transfer(root, e, {"href": e["url"]}))
    endpoint = (
        "https://huggingface.co/datasets/"
        + manifest["repo_id"]
        + ".git/info/lfs/objects/batch"
    )
    headers = {
        "Content-Type": "application/vnd.git-lfs+json",
        "Accept": "application/vnd.git-lfs+json",
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for offset in range(0, len(lfs), 100):
            batch = lfs[offset : offset + 100]
            objects = {e["sha256"]: e for e in batch}
            payload = {
                "operation": "download",
                "transfers": ["basic"],
                "hash_algo": "sha256",
                "objects": [
                    {"oid": oid, "size": e["bytes"]} for oid, e in objects.items()
                ],
            }
            with request(endpoint, json.dumps(payload).encode(), headers) as response:
                actions = json.load(response)["objects"]
            action_map = {o["oid"]: o for o in actions}
            futures = []
            for e in batch:
                obj = action_map.get(e["sha256"], {})
                action = obj.get("actions", {}).get("download")
                if not action:
                    results.append(
                        {
                            "path": e["path"],
                            "status": "failed",
                            "error": "Source batch API supplied no download action",
                        }
                    )
                else:
                    futures.append(pool.submit(transfer, root, e, action))
            results.extend(f.result() for f in concurrent.futures.as_completed(futures))
            print(
                json.dumps(
                    {
                        "lfs_processed": min(offset + 100, len(lfs)),
                        "lfs_pending_total": len(lfs),
                        "failures": sum(r["status"] == "failed" for r in results),
                    }
                ),
                flush=True,
            )
    report = {
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "selected_episodes": len(records),
        "selected_bytes": sum(e["bytes"] for e in entries),
        "files": len(entries),
        "failed": sum(r["status"] == "failed" for r in results),
        "method": "Official Git LFS batch download API; whole-file resume; source SHA256/size checked",
        "results": sorted(results, key=lambda r: r["path"]),
    }
    write_json(BASE / "logs/download_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
    return bool(report["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
