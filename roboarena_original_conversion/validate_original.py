"""Independent source/YAML/NPZ/media checker. Does not import the converter."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import yaml
from original_io import (
    digest,
    positive_int,
    probe_video,
    read_json,
    source_path,
    verify_source,
    write_json,
)
from selection_check import audit_selection
from state_reader import inspect_state

REPO = "RoboArena/DataDump_08-05-2025"
REV = "012901aa47d29f34ccfa8c9472587a97e52a3d62"
REQUIRED = {
    "schema_version",
    "record_revision",
    "trajectory_id",
    "entity_kind",
    "source",
    "family_id",
    "leakage_group_id",
    "task",
    "media",
    "track",
    "outcome",
    "outcome_signature",
    "target_refs",
    "measurement_ref",
    "derivation_ids",
    "cache_refs",
    "quality_flags",
}
BASE_FLAGS = {
    "missing_physical_frame_time",
    "state_video_alignment_unverified",
    "provisional_derivation_family",
    "cross_dataset_overlap_unresolved",
    "unmapped_source_fields",
}


def jsonl(path):
    rows = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        row = json.loads(
            line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
        )
        if type(row) is not dict:
            raise ValueError(f"{path}:{n}: expected JSON object")
        rows.append(row)
    return rows


def validate(out, workers=8, progress=False):
    """Return source-preservation errors and counts for one export directory."""
    out = Path(out).resolve()
    errors, probes, states = [], {}, {}

    def check(condition, message):
        if not condition:
            errors.append(message)

    try:
        manifest, mapping, selection, downloads, summary, context = [
            read_json(out / f)
            for f in (
                "dataset_manifest.json",
                "mapping.json",
                "selection.json",
                "download_manifest.json",
                "summary.json",
                "source_context.json",
            )
        ]
        check(
            all(
                x["repo_id"] == REPO and x["revision"] == REV
                for x in (mapping, selection, downloads, context)
            ),
            "Unrecognized source/revision",
        )
        check(context == manifest["source"], "Source context changed")
        check(
            context["published_viewer_split"]
            == mapping["original_split"]
            == summary["original_split"]
            == "test",
            "Original test split changed",
        )
        check(
            context["training_export"] is False
            and all(
                x["training_approved"] is False for x in (mapping, manifest, summary)
            ),
            "Evaluation source marked for training",
        )
        check(
            mapping["success_decode"] == {"0": False, "1": True},
            "Unsupported Boolean decoding",
        )
        check(
            mapping["duration_units"] == "control_step_index",
            "Unsupported duration meaning",
        )
        for evidence in mapping["duration_evidence"]:
            check(
                digest(out / evidence["local_path"]) == evidence["sha256"],
                "Duration source-code evidence changed",
            )
        for name in ("mapping", "selection", "download_manifest"):
            check(
                digest(out / (name + ".json")) == manifest[name + "_sha256"],
                name + " changed since conversion",
            )
        check(
            digest(out / "upstream/source_inventory.json")
            == manifest["source_inventory_sha256"],
            "Pinned source inventory changed",
        )
        audit_selection(
            selection, downloads, read_json(out / "upstream/source_inventory.json")
        )
        selected = selection["records"]
        check(0 < len(selected) <= 5000, "Record budget exceeded")
        check(
            len(selected) == len({r["sample_id"] for r in selected}),
            "Duplicate source keys",
        )
        entries = {e["path"]: e for e in downloads["files"]}
        check(len(entries) == len(downloads["files"]), "Duplicate download paths")
        expected_paths = {"README.md", "global_metadata.yaml"}
        for r in selected:
            expected_paths.update(
                r[k] for k in ("video_path", "state_path", "metadata_path")
            )
        check(set(entries) == expected_paths, "Source file selection mismatch")
        check(
            sum(e["bytes"] for e in entries.values())
            == downloads["total_bytes"]
            == manifest["source_bytes"],
            "Source byte counts disagree",
        )
        root = out / "assets/roboarena"
        for entry in entries.values():
            verify_source(root, entry)
            check(
                entry["url"]
                == f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"
                + quote(entry["path"], safe="/"),
                "Unpinned source URL",
            )
            if entry["kind"] == "metadata":
                raw = source_path(root, entry["path"]).read_bytes()
                check(
                    hashlib.sha1(
                        b"blob " + str(len(raw)).encode() + b"\0" + raw
                    ).hexdigest()
                    == entry["git_blob_oid"],
                    "Metadata Git object mismatch",
                )
        rows, issues = jsonl(out / "trajectories.jsonl"), jsonl(out / "issues.jsonl")
        issue_keys = {(i["sample_id"], i["severity"], i["code"]) for i in issues}
        with (out / "id_map.csv").open(newline="") as f:
            id_rows = list(csv.DictReader(f))
        statuses = Counter(r["status"] for r in id_rows)
        check(
            set(statuses) <= {"converted", "converted_with_warnings", "skipped"},
            "Unrecognized status",
        )
        check(
            len(selected)
            == len(id_rows)
            == summary["selected_count"]
            == summary["id_map_row_count"]
            == manifest["selected_count"],
            "Selected/status counts disagree",
        )
        check(
            len(rows)
            == summary["written_trajectory_count"]
            == statuses["converted"] + statuses["converted_with_warnings"],
            "Output/status counts disagree",
        )
        for status in ("converted", "converted_with_warnings", "skipped"):
            check(
                statuses[status] == summary[status + "_count"],
                "Status count mismatch: " + status,
            )
        check(
            len(rows) == len({r["trajectory_id"] for r in rows}), "Duplicate output IDs"
        )
        written = [x for x in id_rows if x["status"] != "skipped"]
        skipped_keys = {x["sample_id"] for x in id_rows if x["status"] == "skipped"}
        check(
            [r["trajectory_id"] for r in rows] == [r["trajectory_id"] for r in written],
            "ID map/output order mismatch",
        )
        for r, status in zip(selected, id_rows):
            check(
                r["sample_id"] == status["sample_id"],
                "Selection/status source mismatch",
            )
            if status["status"] == "skipped":
                check(
                    any(
                        i["sample_id"] == r["sample_id"] and i["severity"] == "error"
                        for i in issues
                    ),
                    "Skipped record lacks issue",
                )

        def inspect(r):
            return (
                r["sample_id"],
                probe_video(source_path(root, r["video_path"]), decode=True),
                inspect_state(source_path(root, r["state_path"])),
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(inspect, r): r
                for r in selected
                if r["sample_id"] not in skipped_keys
            }
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    key, video, state = future.result()
                    probes[key], states[key] = video, state
                except Exception as e:  # noqa: BLE001 - Collect validation failures in the report.
                    errors.append(
                        "Source media/state validation failed: "
                        + futures[future]["sample_id"]
                        + ": "
                        + str(e)
                    )
                if progress and (i % 250 == 0 or i == len(futures)):
                    print(
                        json.dumps(
                            {
                                "decoded_episodes": i,
                                "total": len(futures),
                                "errors": len(errors),
                            }
                        ),
                        flush=True,
                    )
        by_key = {r["sample_id"]: r for r in selected}
        flags, successes, completions, cameras = (
            Counter(),
            Counter(),
            Counter(),
            Counter(),
        )
        frames = samples = 0
        for n, row in enumerate(rows, 1):
            prefix = f"record {n}: "
            try:
                check(REQUIRED <= row.keys(), prefix + "missing required field")
                source = row["source"]
                r = by_key[source["sample_id"]]
                with source_path(root, r["metadata_path"]).open() as f:
                    session = yaml.safe_load(f)
                policy = session["policies"][r["policy_id"]]
                tid = "roboarena:" + ":".join(
                    quote(x, safe="")
                    for x in (REPO, REV, r["session_id"], r["policy_id"])
                )
                raw_ref = "assets/roboarena/" + r["metadata_path"]
                policy_ref = raw_ref + "#policies." + r["policy_id"]
                check(
                    row["schema_version"] == "robohue.data.v1.1"
                    and row["record_revision"] == 1,
                    prefix + "schema mismatch",
                )
                check(
                    row["trajectory_id"] == tid and row["entity_kind"] == "execution",
                    prefix + "ID/execution identity mismatch",
                )
                check(
                    source["dataset"] == REPO
                    and source["revision"] == REV
                    and source["original_split"] == "test",
                    prefix + "source/split mismatch",
                )
                check(
                    source["subset"] == "original_collection"
                    and source["raw_record_ref"] == policy_ref,
                    prefix + "source locator mismatch",
                )
                check(
                    source["mapping_version"] == mapping["mapping_version"]
                    and source["split_scope"] == mapping["split_scope"],
                    prefix + "mapping provenance mismatch",
                )
                check(
                    source["execution_id"] == r["sample_id"]
                    and source["execution_id_scope"] == "source_policy_directory",
                    prefix + "execution ID mismatch",
                )
                check(
                    source["session_id"] == r["session_id"]
                    and source["policy_id"] == r["policy_id"]
                    and source["policy_name"] == policy["policy_name"],
                    prefix + "session/policy mismatch",
                )
                expected_dir = f"evaluation_sessions/{r['session_id']}/{r['policy_id']}_{policy['policy_name']}"
                check(
                    r["sample_id"] == expected_dir,
                    prefix + "invalid original policy grouping",
                )
                check(
                    all(
                        r[k].startswith(expected_dir + "/")
                        for k in ("video_path", "state_path")
                    ),
                    prefix + "camera/state not in original episode",
                )
                check(
                    row["task"]
                    == {
                        "task_id": None,
                        "instruction": session["language_instruction"],
                    },
                    prefix + "instruction changed or task ID invented",
                )
                check(
                    source["evaluator_name"] == session["evaluator_name"]
                    and source["evaluation_location"] == session["evaluation_location"],
                    prefix + "attribution changed",
                )
                check(
                    source["session_timestamps"]
                    == {
                        "created_at": session["session_creation_timestamp"],
                        "completed_at": session["session_completion_timestamp"],
                    },
                    prefix + "session times changed",
                )
                check(
                    source["policy_duration_raw"] == policy["duration"]
                    and source["policy_duration_units"] == "control_step_index",
                    prefix + "duration value/meaning changed",
                )
                check(
                    type(policy["binary_success"]) is int
                    and policy["binary_success"] in (0, 1),
                    prefix + "unrecognized source success",
                )
                check(
                    type(row["outcome"]["success"]) is bool
                    and int(row["outcome"]["success"]) == policy["binary_success"],
                    prefix + "explicit success changed",
                )
                check(
                    source["binary_success_raw"] == policy["binary_success"]
                    and source["success_evidence_ref"] == policy_ref + ".binary_success"
                    and source["label_source_ref"] == raw_ref + "#evaluator_name",
                    prefix + "label evidence mismatch",
                )
                check(
                    row["outcome"]["quality_label"] is None,
                    prefix + "quality category invented",
                )
                score = row["outcome"]["partial_success"]
                check(
                    score
                    == {
                        "value": policy["partial_success"],
                        "scope": "source_policy_episode",
                        "semantics": "source_task_completion_score_0_to_1",
                        "label_source": session["evaluator_name"],
                        "evidence_ref": policy_ref + ".partial_success",
                    },
                    prefix + "completion changed",
                )
                check(
                    type(policy["partial_success"]) in (int, float)
                    and 0 <= policy["partial_success"] <= 1,
                    prefix + "source score range/type mismatch",
                )
                check(row["family_id"] == tid, prefix + "family ID changed")
                check(
                    row["leakage_group_id"]
                    == "roboarena-session:"
                    + ":".join(quote(x, safe="") for x in (REPO, REV, r["session_id"])),
                    prefix + "session leakage grouping mismatch",
                )
                check(len(row["media"]) == 1, prefix + "unexpected active camera count")
                media, probe, state = (
                    row["media"][0],
                    probes[r["sample_id"]],
                    states[r["sample_id"]],
                )
                check(
                    type(policy["duration"]) is int
                    and policy["duration"] == state["sample_count"] - 1,
                    prefix + "duration/control-step relation mismatch",
                )
                check(
                    media["path"] == "assets/roboarena/" + r["video_path"]
                    and media["kind"] == "video",
                    prefix + "video reference mismatch",
                )
                check(
                    media["camera_id"] == r["camera_id"]
                    and r["video_path"].endswith(
                        "_video_" + media["camera_id"] + ".mp4"
                    ),
                    prefix + "camera identity mismatch",
                )
                check(
                    media["frame_count"] == probe["frame_count"]
                    and media["frame_order"]
                    == {"start": 0, "stop_exclusive": probe["frame_count"], "step": 1},
                    prefix + "source frame sequence changed",
                )
                check(
                    media["encoded_rate"] == probe["r_frame_rate"]
                    and media["width"] == probe["width"]
                    and media["height"] == probe["height"],
                    prefix + "encoded properties mismatch",
                )
                check(
                    media["timebase"] == "encoded_frame_index"
                    and media["timestamps_s"] is None
                    and media["sync_ref"] is None,
                    prefix + "physical time/sync invented",
                )
                check(
                    row["track"]
                    == {
                        **state,
                        "path": "assets/roboarena/" + r["state_path"],
                        "kind": "source_state_and_actions",
                        "timestamps_s": None,
                        "alignment_to_video": None,
                        "units": None,
                    },
                    prefix + "state reference/dimensions altered or timing invented",
                )
                others = source["other_camera_refs"]
                check(
                    others
                    == [
                        {
                            "source_path": p,
                            "url": f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/"
                            + quote(p, safe="/"),
                            "downloaded_for_this_record": False,
                        }
                        for p in r["other_camera_paths"]
                    ],
                    prefix + "other camera references mismatch",
                )
                unmapped = sorted(
                    set(session)
                    - {
                        "language_instruction",
                        "policies",
                        "evaluator_name",
                        "evaluation_location",
                        "session_creation_timestamp",
                        "session_completion_timestamp",
                    }
                )
                unmapped += [
                    "policies." + r["policy_id"] + "." + k
                    for k in sorted(
                        set(policy)
                        - {
                            "policy_name",
                            "binary_success",
                            "partial_success",
                            "duration",
                        }
                    )
                ]
                check(
                    source["unmapped_fields"] == unmapped,
                    prefix + "unmapped source keys lost",
                )
                check(
                    all(
                        row[k] is None for k in ("measurement_ref", "outcome_signature")
                    ),
                    prefix + "unsupported measurement/signature invented",
                )
                check(
                    all(
                        row[k] == []
                        for k in ("target_refs", "derivation_ids", "cache_refs")
                    ),
                    prefix + "unsupported derived references",
                )
                expected_flags = BASE_FLAGS | (
                    {"nonfinite_source_state_values"}
                    if state["nonfinite_value_count"]
                    else set()
                )
                check(
                    set(row["quality_flags"]) == expected_flags,
                    prefix + "missing/extra uncertainty flags",
                )
                for code in row["quality_flags"]:
                    check(
                        (r["sample_id"], "warning", code) in issue_keys,
                        prefix + "warning lacks issue: " + code,
                    )
                flags.update(row["quality_flags"])
                frames += probe["frame_count"]
                samples += state["sample_count"]
                successes[str(row["outcome"]["success"]).lower()] += 1
                completions[str(score["value"])] += 1
                cameras[media["camera_id"]] += 1
            except Exception as e:  # noqa: BLE001 - Collect validation failures in the report.
                errors.append(prefix + type(e).__name__ + ": " + str(e))
        check(dict(flags) == summary["warnings_by_code"], "Warning counts disagree")
        check(
            frames == summary["frame_count"]
            and samples == summary["state_sample_count"],
            "Frame/state counts disagree",
        )
        check(
            dict(successes) == summary["success_counts"]
            and dict(completions) == summary["completion_counts"]
            and dict(cameras) == summary["camera_counts"],
            "Label/camera counts disagree",
        )
        check(
            summary["quality_category_counts"] == {"missing": len(rows)}
            and summary["source_subset_counts"] == {"original_collection": len(rows)},
            "Subset/missing-quality counts disagree",
        )
    except Exception as e:  # noqa: BLE001 - Collect validation failures in the report.
        errors.append(type(e).__name__ + ": " + str(e))
    return {
        "passed": not errors,
        "errors": errors,
        "record_count": len(rows) if "rows" in locals() else 0,
        "selected_count": len(selected) if "selected" in locals() else 0,
        "video_files_decoded": len(probes),
        "state_files_read": len(states),
        "source_files_hash_checked": len(entries)
        if "entries" in locals() and not errors
        else None,
        "referenced_frames_checked": frames if "frames" in locals() else 0,
        "state_samples_checked": samples if "samples" in locals() else 0,
        "warnings_by_code": dict(flags) if "flags" in locals() else {},
        "training_approved": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="export directory to check")
    parser.add_argument(
        "--workers", type=positive_int, default=8, help="parallel workers (default: 8)"
    )
    args = parser.parse_args()
    report = validate(args.output, args.workers, progress=True)
    write_json(args.output / "validation_report.json", report)
    print(json.dumps(report), flush=True)
    raise SystemExit(not report["passed"])
