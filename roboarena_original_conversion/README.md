# Original RoboArena conversion

The mentor has reviewed and approved this conversion output. This is the active
package: **4,613 episodes from 796 sessions**, using schema `robohue.data.v1.1`.

Read the [Task A usage guide](../RoboHue_Task_A_RBM_Conversion.md) for installation,
reading records, command options, full conversion runs, output fields and
troubleshooting. Start with the [root README](../README.md) for a quick setup.

## Run a small conversion

From the parent `robohue/` directory, after installing dependencies and FFmpeg:

```sh
.venv/bin/python roboarena_original_conversion/download_original.py --limit 10
.venv/bin/python roboarena_original_conversion/convert_original.py --out exports/first10 --limit 10
.venv/bin/python roboarena_original_conversion/validate_original.py exports/first10
```

Omit `--limit` from download and conversion commands for the full selection.
Use a fresh output directory for each run. Export media paths are relative to that
export's directory; new exports link to this package's `assets/` and `upstream/`.

## Files to inspect

| File | Purpose |
|---|---|
| `trajectories.jsonl` | Generated episode index |
| `mapping.json` | Exact field mapping and label decoding |
| `field_availability.json` | Explanation of absent and unresolved fields |
| `summary.json` | Generated counts and warning totals |
| `validation_report.json` | Result of checking source, records, videos and state arrays |
| `selection.json`, `source_context.json`, `download_manifest.json` | Pinned source configuration |
| `upstream/` | Required source inventory, duration-rule evidence and licenses |

The main scripts are `download_original.py`, `convert_original.py` and
`validate_original.py`. They use `original_io.py`, `selection_check.py` and
`state_reader.py`. No archived tests or demos are needed to run them.

The recorded source split remains `test`, with `training_approved: false`.
Physical timing, state/video alignment and state units remain unresolved; see the
usage guide for their meaning. Conversion approval preserves that source metadata.
