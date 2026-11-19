# RoboHue data conversion

The active program is in [`roboarena_original_conversion/`](roboarena_original_conversion/README.md).
The existing export contains **4,613 episodes from 796 sessions**. To use it now,
read `roboarena_original_conversion/trajectories.jsonl`; resolve its relative media
and state paths from that same directory.

## Setup and quick start

Run commands from this directory. Use Python 3.11 or newer and install FFmpeg
(which also supplies `ffprobe`; on macOS, `brew install ffmpeg`).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r roboarena_original_conversion/requirements.txt
ffmpeg -version
ffprobe -version

# Download or verify the source files for the first ten selected episodes.
.venv/bin/python roboarena_original_conversion/download_original.py --limit 10

# Write a ten-episode export in a fresh directory, then validate it.
.venv/bin/python roboarena_original_conversion/convert_original.py --out exports/first10 --limit 10
.venv/bin/python roboarena_original_conversion/validate_original.py exports/first10
```

For the full dataset, run the downloader without `--limit`, then convert to a new
output directory without `--limit`. The [Task A usage guide](RoboHue_Task_A_RBM_Conversion.md)
explains full runs, output fields, validation, troubleshooting and moving exports.
No model, API key or training run is needed.

## What belongs to the program

- `roboarena_original_conversion/*.py`: downloader, converter, validator and their helpers.
- `requirements.txt` inside that directory: Python dependencies.
- `mapping.json`, `selection.json`, `source_context.json`, `download_manifest.json`
  and `upstream/` inside that directory: pinned configuration and validation evidence.
- `field_availability.json`: explanation of fields that are absent or unresolved.