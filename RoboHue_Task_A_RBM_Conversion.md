# RoboHue — Task A: Conversion usage guide

This package converts a fixed selection from the original RoboArena release into
`robohue.data.v1.1` records. It preserves source instructions, explicit success and
completion labels, source IDs, original video frame order and state/action arrays.
The mentor has reviewed and approved the conversion output.

The source is `RoboArena/DataDump_08-05-2025`, pinned to revision
`012901aa47d29f34ccfa8c9472587a97e52a3d62`. There are **4,613 policy episodes from
796 sessions**, with one selected original camera video and one state/action file
per episode. The selected source files occupy about 2.48 GB (2.3 GiB).
The recorded source split remains `test`; approval of the conversion does not
change that split or the package's `training_approved: false` metadata.

This is the usage guide for the approved conversion, replacing the original
assignment checklist. The active package is `roboarena_original_conversion/`;
older RBM experiments remain under the ignored `archive/` directory.

## 1. Install dependencies

Run every command below from the repository's root directory, the directory
containing the top-level `README.md` and `roboarena_original_conversion/`.
Use Python 3.11 or newer. The existing local environment uses Python 3.14.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r roboarena_original_conversion/requirements.txt
```

Install FFmpeg separately. On macOS with Homebrew:

```sh
brew install ffmpeg
```

Both commands must be available on `PATH`:

```sh
ffmpeg -version
ffprobe -version
```

The Python dependencies are NumPy and PyYAML. No Hugging Face library, model,
API key, Pillow or OpenCV is needed for the core program.

## 2. Read the existing export

If the generated output and source assets are already present, no conversion is
necessary. `trajectories.jsonl` contains one JSON object per episode. Paths inside
a record are relative to the **export directory**, not the current working directory.

```python
import json
from pathlib import Path

export = Path("roboarena_original_conversion")
with (export / "trajectories.jsonl").open(encoding="utf-8") as stream:
    episode = json.loads(next(stream))

print(episode["trajectory_id"])
print(episode["task"]["instruction"])
print(episode["outcome"]["success"])
print(episode["outcome"]["partial_success"]["value"])
print("Video:", export / episode["media"][0]["path"])
print("State/actions:", export / episode["track"]["path"])
```

Use `state_reader.read_source_state(path)` from this package to read the original
NPZ object-array format. The converter keeps those arrays in their source files;
it does not generate a separate numeric export. State and video sample counts can
differ, and their alignment is unresolved.

## 3. Download → convert → validate

### Try ten episodes

```sh
.venv/bin/python roboarena_original_conversion/download_original.py --limit 10
.venv/bin/python roboarena_original_conversion/convert_original.py --out exports/first10 --limit 10
.venv/bin/python roboarena_original_conversion/validate_original.py exports/first10
```

`--limit` takes the first N entries of the fixed selection, in its existing order.
It does not sample randomly. Give the downloader at least the same limit as the
converter, or download all source files first.

### Rebuild the full export

```sh
.venv/bin/python roboarena_original_conversion/download_original.py
.venv/bin/python roboarena_original_conversion/convert_original.py --out exports/full
.venv/bin/python roboarena_original_conversion/validate_original.py exports/full
```

The downloader reuses files whose size and SHA-256 already match. Network access
is needed for missing or changed source files. Conversion and validation use local
files and can run offline once downloading is complete.

Choose a fresh output directory for each conversion. Without `--out`, the converter
targets this package directory. It refuses to replace an existing
`trajectories.jsonl` unless `--overwrite` is explicit. Overwrite replaces generated
conversion files; using a fresh directory makes runs easier to compare.

### Command options

| Script | Arguments |
|---|---|
| `download_original.py` | `--limit N`, `--workers N` (default 12) |
| `convert_original.py` | `--out DIR` (default package directory), `--limit N`, `--workers N` (default 8), `--overwrite` |
| `validate_original.py` | Export directory (required), `--workers N` (default 8) |
| `selection_check.py` | Optional package/export directory (default package directory); checks the selection against the source inventory without decoding media |

All scripts support `--help`. Limits and worker counts must be positive integers.
Reduce `--workers` if your machine is busy or memory is limited.

## 4. Understand the outputs

| Generated file | Purpose |
|---|---|
| `trajectories.jsonl` | Main index: one episode per line |
| `id_map.csv` | Original locator, output ID and conversion status for every selected episode |
| `issues.jsonl` | Explained warnings and conversion failures |
| `summary.json` | Selected/written/skipped counts, labels, frames and state samples |
| `dataset_manifest.json` | Source identity and hashes of conversion inputs |
| `examples/before_after.json` | Up to ten source/output pairs for optional inspection |
| `validation_report.json` | Validator result; written by the validation command |

A separate export also contains copies of `mapping.json`, `source_context.json`,
`selection.json` and `download_manifest.json`, trimmed to the selected records
where applicable. Its `assets/` and `upstream/` entries are symlinks to this package.
Keep those targets available. To move an export to another machine, copy the
linked directories' contents into the export rather than copying only the links.
The validator requires `upstream/` as well as the source assets.

### What each record means

- `trajectory_id`: stable source identity in the form
  `roboarena:{repository}:{revision}:{session_id}:{policy_id}`, with URL-encoded components.
- `source`: original dataset, revision, session/policy IDs, metadata locator,
  source split and attribution. Original YAML preserves unmapped metadata.
- `task.instruction`: unchanged source text. `task.task_id` is null.
- `media[0]`: selected video, camera identity, frame count and encoded frame order.
  Other camera URLs remain under `source.other_camera_refs`; they are not downloaded.
- `track`: original state/action reference, sample count and field dimensions.
- `outcome.success`: Boolean decoded from explicit source `binary_success` 0/1.
- `outcome.partial_success.value`: unchanged source completion score in [0, 1].
- `quality_flags`: unresolved timing, alignment, grouping and metadata limitations.

Physical timestamps, state units and video/state alignment remain unknown. Raw
`duration` is a zero-based control-step index, not seconds. No progress curves,
preferences, measurements or new behavior labels are generated. See `mapping.json`
for exact rules and `field_availability.json` in this package for null/empty fields.

## 5. Check a run

The converter exits with status 1 if any episode was skipped. Inspect
`issues.jsonl` and `id_map.csv` to find its source locator and failure reason;
failed records are not silently replaced. Warnings are recorded separately.

The validator compares IDs, instructions, labels, counts, file hashes, state
shapes, frame order and provenance against the pinned source. It fully decodes
selected videos and reads the state files. It writes `validation_report.json`,
prints the report and exits with status 0 on success or 1 on failure. Full
validation processes all selected videos and takes longer than a metadata check.

The existing full-run report records **4,613 decoded videos, 1,465,289 frames and
1,461,519 state samples**, with zero structural errors. The conversion summary
records zero skipped episodes. Warnings remain visible even when validation passes.

| Problem | Action |
|---|---|
| Missing NumPy or YAML module | Install `requirements.txt` with the same Python used to run the scripts |
| `ffmpeg` or `ffprobe` not found | Install FFmpeg and check both commands on `PATH` |
| `Source missing/changed` | Run the downloader to restore and verify the selected source files |
| Output already exists | Choose a fresh `--out` directory, or explicitly use `--overwrite` |
| Broken `assets/` or `upstream/` link | Restore its target, or copy the actual target contents into the export |
| Manifest/hash mismatch | Restore the pinned inputs and convert into a fresh directory; investigate unexpected changes |
| Validation passes with warnings | Read the issue log; a passing structural check does not resolve missing timing or alignment |
