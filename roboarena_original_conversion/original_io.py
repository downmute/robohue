"""Helpers for the pinned original RoboArena collection; no RBM-ID matching."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import yaml


def positive_int(value):
    """Parse a positive CLI limit or worker count."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_path(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError("Source locator must be a relative path inside the collection")
    return Path(root).joinpath(*p.parts)


def verify_source(root, entry):
    p = source_path(root, entry["path"])
    if (
        not p.is_file()
        or p.stat().st_size != entry["bytes"]
        or digest(p) != entry["sha256"]
    ):
        raise ValueError("Source missing/changed: " + entry["path"])
    return p


def read_session(root, record):
    with source_path(root, record["metadata_path"]).open(encoding="utf-8") as f:
        session = yaml.safe_load(f)
    policy = session["policies"][record["policy_id"]]
    if policy["policy_name"] != record["policy_name"]:
        raise ValueError("Policy metadata does not match its source directory")
    return session, policy


def episode_id(context, record):
    return "roboarena:" + ":".join(
        quote(str(v), safe="")
        for v in (
            context["repo_id"],
            context["revision"],
            record["session_id"],
            record["policy_id"],
        )
    )


def metadata_ref(record, field=None):
    base = "assets/roboarena/" + record["metadata_path"]
    return (
        base + "#" + (field if field is not None else "policies." + record["policy_id"])
    )


def selected_files(manifest, records):
    paths = {"README.md", "global_metadata.yaml"}
    for row in records:
        paths.update(row[k] for k in ("metadata_path", "video_path", "state_path"))
    return [e for e in manifest["files"] if e["path"] in paths]


def probe_video(path, decode=False):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads",
            "1",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stderr.strip():
        raise ValueError(result.stderr.strip())
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError("Expected one selected original camera stream")
    probe = streams[0]
    probe["frame_count"] = int(probe["nb_read_frames"])
    if probe["frame_count"] <= 0:
        raise ValueError("No decoded source frames")
    if decode:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-threads",
                "1",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    return probe
