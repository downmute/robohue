"""Convert explicit original RoboArena policy episodes, without RBM UUID matching."""

import argparse
import concurrent.futures
import csv
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from original_io import (
    digest,
    episode_id,
    metadata_ref,
    positive_int,
    probe_video,
    read_json,
    read_session,
    selected_files,
    source_path,
    verify_source,
    write_json,
)
from selection_check import audit_selection
from state_reader import inspect_state

BASE = Path(__file__).resolve().parent
FLAGS = {
    "missing_physical_frame_time": "Session wall-clock times exist; per-frame physical capture times do not.",
    "state_video_alignment_unverified": "Original state and camera share an episode folder; sample-to-frame timing is not established.",
    "provisional_derivation_family": "Self family is a placeholder; no cross-dataset identity mapping supplied.",
    "cross_dataset_overlap_unresolved": "Original session/policy locators exported; relation to regenerated RBM UUIDs not asserted.",
    "unmapped_source_fields": "Original preference, feedback and any extra metadata retained in source YAML, without creating feedback records.",
}


def convert_record(record, context, mapping, root, file_map):
    """Preserve one original policy episode after checking its source files."""
    if (
        mapping["success_decode"] != {"0": False, "1": True}
        or mapping["training_approved"] is not False
    ):
        raise ValueError("Unsupported source label/split mapping")
    if (
        mapping["repo_id"] != context["repo_id"]
        or mapping["revision"] != context["revision"]
    ):
        raise ValueError("Mapping/source revision mismatch")
    if (
        mapping["original_split"] != context["published_viewer_split"]
        or context["training_export"] is not False
    ):
        raise ValueError("Original published source role must be preserved")
    for key in ("metadata_path", "video_path", "state_path"):
        verify_source(root, file_map[record[key]])
    session, policy = read_session(root, record)
    success, completion = policy["binary_success"], policy["partial_success"]
    if type(success) is not int or success not in (0, 1):
        raise ValueError("Explicit binary_success must be source integer 0 or 1")
    if type(completion) not in (int, float) or not 0 <= completion <= 1:
        raise ValueError("Source completion is not a supported [0,1] annotation")
    if type(session["language_instruction"]) is not str:
        raise ValueError("Instruction is not source text")
    prefix = record["sample_id"] + "/"
    if (
        prefix
        != f"evaluation_sessions/{record['session_id']}/{record['policy_id']}_{policy['policy_name']}/"
    ):
        raise ValueError(
            "Sample identity differs from original session/policy directory"
        )
    if any(not record[k].startswith(prefix) for k in ("video_path", "state_path")):
        raise ValueError(
            "Camera/state not explicitly grouped in original policy directory"
        )
    if not record["video_path"].endswith("_video_" + record["camera_id"] + ".mp4"):
        raise ValueError("Camera ID differs from original filename")
    probe = probe_video(source_path(root, record["video_path"]))
    track = inspect_state(source_path(root, record["state_path"]))
    if (
        mapping.get("duration_units") != "control_step_index"
        or track["sample_count"] != policy["duration"] + 1
    ):
        raise ValueError(
            "Duration does not match the verified zero-based control-step rule"
        )
    tid = episode_id(context, record)
    policy_key = "policies." + record["policy_id"]
    mapped_session = {
        "language_instruction",
        "policies",
        "evaluator_name",
        "evaluation_location",
        "session_creation_timestamp",
        "session_completion_timestamp",
    }
    mapped_policy = {"policy_name", "binary_success", "partial_success", "duration"}
    source = {
        "dataset": context["repo_id"],
        "revision": context["revision"],
        "subset": "original_collection",
        "sample_id": record["sample_id"],
        "session_id": record["session_id"],
        "policy_id": record["policy_id"],
        "policy_name": policy["policy_name"],
        "execution_id": record["sample_id"],
        "execution_id_scope": "source_policy_directory",
        "original_split": mapping["original_split"],
        "split_scope": mapping["split_scope"],
        "raw_record_ref": metadata_ref(record),
        "mapping_version": mapping["mapping_version"],
        "evaluator_name": session["evaluator_name"],
        "evaluation_location": session["evaluation_location"],
        "session_timestamps": {
            "created_at": session["session_creation_timestamp"],
            "completed_at": session["session_completion_timestamp"],
        },
        "policy_duration_raw": policy["duration"],
        "policy_duration_units": "control_step_index",
        "binary_success_raw": success,
        "success_evidence_ref": metadata_ref(record, policy_key + ".binary_success"),
        "label_source_ref": metadata_ref(record, "evaluator_name"),
        "unmapped_fields": sorted(set(session) - mapped_session)
        + [policy_key + "." + k for k in sorted(set(policy) - mapped_policy)],
        "other_camera_refs": [
            {
                "source_path": p,
                "url": "https://huggingface.co/datasets/"
                + context["repo_id"]
                + "/resolve/"
                + context["revision"]
                + "/"
                + quote(p, safe="/"),
                "downloaded_for_this_record": False,
            }
            for p in record["other_camera_paths"]
        ],
    }
    track.update(
        {
            "path": "assets/roboarena/" + record["state_path"],
            "kind": "source_state_and_actions",
            "timestamps_s": None,
            "alignment_to_video": None,
            "units": None,
        }
    )
    flags = list(FLAGS)
    if track["nonfinite_value_count"]:
        flags.append("nonfinite_source_state_values")
    return {
        "schema_version": mapping["schema_version"],
        "record_revision": 1,
        "trajectory_id": tid,
        "entity_kind": "execution",
        "source": source,
        "family_id": tid,
        "leakage_group_id": "roboarena-session:"
        + ":".join(
            quote(v, safe="")
            for v in (context["repo_id"], context["revision"], record["session_id"])
        ),
        "task": {"task_id": None, "instruction": session["language_instruction"]},
        "media": [
            {
                "kind": "video",
                "camera_id": record["camera_id"],
                "path": "assets/roboarena/" + record["video_path"],
                "frame_count": probe["frame_count"],
                "frame_order": {
                    "start": 0,
                    "stop_exclusive": probe["frame_count"],
                    "step": 1,
                },
                "timebase": "encoded_frame_index",
                "timestamps_s": None,
                "sync_ref": None,
                "encoded_rate": probe["r_frame_rate"],
                "width": probe["width"],
                "height": probe["height"],
            }
        ],
        "track": track,
        "outcome": {
            "success": mapping["success_decode"][str(success)],
            "quality_label": None,
            "partial_success": {
                "value": completion,
                "scope": "source_policy_episode",
                "semantics": "source_task_completion_score_0_to_1",
                "label_source": session["evaluator_name"],
                "evidence_ref": metadata_ref(record, policy_key + ".partial_success"),
            },
        },
        "outcome_signature": None,
        "target_refs": [],
        "measurement_ref": None,
        "derivation_ids": [],
        "cache_refs": [],
        "quality_flags": flags,
    }


def run(out, limit=None, workers=8, overwrite=False):
    """Write a deterministic export using the pinned package inputs and assets."""
    out = Path(out).resolve()
    if (out / "trajectories.jsonl").exists() and not overwrite:
        raise FileExistsError(
            "Output exists; choose a new directory or explicitly use --overwrite"
        )
    context, mapping = (
        read_json(BASE / "source_context.json"),
        read_json(BASE / "mapping.json"),
    )
    selection, downloads = (
        read_json(BASE / "selection.json"),
        read_json(BASE / "download_manifest.json"),
    )
    audit_selection(
        selection, downloads, read_json(BASE / "upstream/source_inventory.json")
    )
    records = selection["records"][:limit] if limit else selection["records"]
    if len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Duplicate original sample locators; no outputs overwritten")
    if len({episode_id(context, r) for r in records}) != len(records):
        raise ValueError("Conflicting output identities")
    file_map = {e["path"]: e for e in downloads["files"]}
    root = BASE / "assets/roboarena"
    out.mkdir(parents=True, exist_ok=True)
    if out != BASE:
        if not (out / "assets").exists():
            (out / "assets").symlink_to(
                os.path.relpath(BASE / "assets", out), target_is_directory=True
            )
        if not (out / "upstream").exists():
            (out / "upstream").symlink_to(
                os.path.relpath(BASE / "upstream", out), target_is_directory=True
            )
        write_json(out / "mapping.json", mapping)
        write_json(out / "source_context.json", context)
        write_json(
            out / "selection.json",
            {**selection, "records": records, "selected_count": len(records)},
        )
        files = selected_files(downloads, records)
        write_json(
            out / "download_manifest.json",
            {
                **downloads,
                "files": files,
                "selected_policy_episodes": len(records),
                "total_bytes": sum(e["bytes"] for e in files),
            },
        )

    def attempt(record):
        try:
            return convert_record(record, context, mapping, root, file_map), None
        except Exception as e:  # noqa: BLE001 - Record failures belong in the issue log; continue the batch.
            return None, type(e).__name__ + ": " + str(e)

    examples, count, skipped, frames, samples = [], 0, 0, 0, 0
    label_counts, completion_counts, camera_counts, warnings = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    with (
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
        (out / "trajectories.jsonl").open("w") as trajectories,
        (out / "issues.jsonl").open("w") as issues,
        (out / "id_map.csv").open("w", newline="") as ids,
    ):
        writer = csv.DictWriter(
            ids, fieldnames=["sample_id", "raw_record_ref", "trajectory_id", "status"]
        )
        writer.writeheader()
        for i, (r, (row, error)) in enumerate(
            zip(records, pool.map(attempt, records)), 1
        ):
            status = "skipped" if error else "converted_with_warnings"
            writer.writerow(
                {
                    "sample_id": r["sample_id"],
                    "raw_record_ref": metadata_ref(r),
                    "trajectory_id": episode_id(context, r),
                    "status": status,
                }
            )
            if error:
                skipped += 1
                issues.write(
                    json.dumps(
                        {
                            "sample_id": r["sample_id"],
                            "source_locator": metadata_ref(r),
                            "severity": "error",
                            "code": "conversion_failed",
                            "observed": error,
                            "action": "skipped_without_replacement",
                        }
                    )
                    + "\n"
                )
                continue
            trajectories.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            )
            count += 1
            frames += row["media"][0]["frame_count"]
            samples += row["track"]["sample_count"]
            label_counts[str(row["outcome"]["success"]).lower()] += 1
            completion_counts[str(row["outcome"]["partial_success"]["value"])] += 1
            camera_counts[r["camera_id"]] += 1
            for code in row["quality_flags"]:
                warnings[code] += 1
                issues.write(
                    json.dumps(
                        {
                            "sample_id": r["sample_id"],
                            "source_locator": metadata_ref(r),
                            "severity": "warning",
                            "code": code,
                            "action": FLAGS.get(
                                code,
                                "Original state nonfinite values preserved; no clipping",
                            ),
                        }
                    )
                    + "\n"
                )
            if len(examples) < 10:
                session, policy = read_session(root, r)
                examples.append(
                    {
                        "before": {
                            "sample_locator": r,
                            "session_metadata": session,
                            "policy_metadata": policy,
                        },
                        "after": row,
                    }
                )
            if i % 250 == 0:
                print(
                    json.dumps({"processed": i, "written": count, "skipped": skipped}),
                    flush=True,
                )
    summary = {
        "selected_count": len(records),
        "converted_count": 0,
        "converted_with_warnings_count": count,
        "skipped_count": skipped,
        "written_trajectory_count": count,
        "id_map_row_count": len(records),
        "frame_count": frames,
        "state_sample_count": samples,
        "success_counts": dict(label_counts),
        "completion_counts": dict(completion_counts),
        "camera_counts": dict(camera_counts),
        "source_subset_counts": {"original_collection": count},
        "quality_category_counts": {"missing": count},
        "warnings_by_code": dict(warnings),
        "training_approved": False,
        "original_split": "test",
    }
    write_json(out / "summary.json", summary)
    write_json(out / "examples/before_after.json", examples)
    write_json(
        out / "dataset_manifest.json",
        {
            "source": context,
            "mapping_version": mapping["mapping_version"],
            "mapping_sha256": digest(out / "mapping.json"),
            "selection_sha256": digest(out / "selection.json"),
            "download_manifest_sha256": digest(out / "download_manifest.json"),
            "selected_count": len(records),
            "source_inventory_sha256": digest(out / "upstream/source_inventory.json"),
            "source_bytes": sum(e["bytes"] for e in selected_files(downloads, records)),
            "training_approved": False,
            "command": f"convert_original.py --out {out} --limit {len(records)} --workers {workers}",
        },
    )
    print(json.dumps(summary), flush=True)
    return bool(skipped)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=BASE,
        help="export directory (default: package directory)",
    )
    p.add_argument(
        "--limit",
        type=positive_int,
        help="convert/download the first N selected episodes",
    )
    p.add_argument(
        "--workers", type=positive_int, default=8, help="parallel workers (default: 8)"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="replace existing conversion output"
    )
    a = p.parse_args()
    raise SystemExit(run(a.out, a.limit, a.workers, a.overwrite))
