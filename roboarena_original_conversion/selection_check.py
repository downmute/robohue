"""Verify the fixed episode/camera selection against the pinned full source tree."""

import argparse
from collections import defaultdict
from pathlib import Path

from original_io import read_json


def audit_selection(selection, downloads, inventory):
    files = {e["path"]: e for e in inventory if e["type"] == "file"}
    groups = defaultdict(list)
    for p in files:
        parts = p.split("/")
        if len(parts) == 4 and parts[0] == "evaluation_sessions":
            groups["/".join(parts[:3])].append(p)
    if len(groups) != 4613:
        raise ValueError("Unexpected original source episode count")
    if len(selection["records"]) == 4613 and {
        r["sample_id"] for r in selection["records"]
    } != set(groups):
        raise ValueError("Full selection does not cover every original episode")
    wanted = {"README.md", "global_metadata.yaml"}
    for r in selection["records"]:
        paths = groups[r["sample_id"]]
        videos = sorted(p for p in paths if p.endswith(".mp4"))
        states = [p for p in paths if p.endswith(".npz")]
        primary = None
        for camera in ("left", "right", "wrist"):
            candidates = [p for p in videos if p.endswith("_video_" + camera + ".mp4")]
            if candidates:
                if len(candidates) != 1:
                    raise ValueError("Ambiguous source camera")
                primary = candidates[0]
                break
        if (
            r["video_path"] != primary
            or r["camera_id"] != camera
            or states != [r["state_path"]]
        ):
            raise ValueError(
                "Primary camera/state rule differs from original inventory"
            )
        if r["other_camera_paths"] != [p for p in videos if p != primary]:
            raise ValueError("Original additional cameras lost")
        if r["metadata_path"] != f"evaluation_sessions/{r['session_id']}/metadata.yaml":
            raise ValueError("Session metadata locator changed")
        wanted.update(r[k] for k in ("video_path", "state_path", "metadata_path"))
    if {e["path"] for e in downloads["files"]} != wanted:
        raise ValueError("Download manifest differs from fixed episode selection")
    for e in downloads["files"]:
        src = files[e["path"]]
        if src["size"] != e["bytes"]:
            raise ValueError("Source inventory size mismatch")
        if "lfs" in src:
            if src["lfs"]["oid"] != e["sha256"]:
                raise ValueError("Pinned source LFS digest mismatch")
        elif src["oid"] != e["git_blob_oid"]:
            raise ValueError("Pinned source Git blob mismatch")
    return {
        "passed": True,
        "source_policy_episodes": len(groups),
        "selected_episodes": len(selection["records"]),
        "source_camera_videos": sum(p.endswith(".mp4") for p in files),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "package", type=Path, nargs="?", default=Path(__file__).resolve().parent
    )
    a = p.parse_args().package
    print(
        audit_selection(
            read_json(a / "selection.json"),
            read_json(a / "download_manifest.json"),
            read_json(a / "upstream/source_inventory.json"),
        )
    )
