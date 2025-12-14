# Target Definition Integration - Update Summary

## Changes Made

Updated `run_scheduler.py` to support complete end-to-end pipeline execution from target definition files, enabling generation of target manifests and visibility catalogs.

## Key Features

### 1. Target Definition File Support

The script accepts a base directory containing target definition files (JSON format):

```bash
--target-definitions /Users/tsbarcl2/gitcode/PandoraTargetList/target_definition_files
```

### 2. Automatic Manifest Generation

When `--target-definitions` is provided, the script:
- Generates `*_targets.csv` manifests in `output/data/` (or whatever directory you pass via `--output`)
- Creates manifests for all 4 categories:
  - `exoplanet_targets.csv` (primary targets)
  - `auxiliary-standard_targets.csv`
  - `monitoring-standard_targets.csv`
  - `occultation-standard_targets.csv`

### 3. Visibility Catalog Generation

With both `--target-definitions` and `--generate-visibility`:
- Generates visibility time series for all targets
- Stores in `output/data/targets/` (primary) and `output/data/aux_targets/` (auxiliary) (under your chosen `--output` directory)
- Uses GMAT ephemeris for orbit calculations
- Applies configurable avoidance angles (sun, moon, earth)

### 4. Three Usage Modes

#### Mode 1: Full Pipeline (Generate Everything)
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --target-definitions /path/to/target_definition_files \
    --generate-visibility \
    --gmat-ephemeris /path/to/ephemeris.txt \
    --show-progress
```
Generates: Manifests → Visibility → Schedule → XML

#### Mode 2: Generate Manifests Only
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --target-definitions /path/to/target_definition_files
```
Generates: Manifests → Schedule → XML (uses existing or on-demand visibility)

#### Mode 3: Use Existing Manifests (No Generation)
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output
```
Uses: Existing manifests and visibility files already present under `output/data/` (or whatever directory you pass via `--output`).

## Technical Implementation

### Configuration Flow

```python
# In main()
target_def_base = args.target_definitions

if args.generate_visibility and not target_def_base:
    logger.error("Visibility generation requires target definitions. "
                 "Please provide target definitions via --target-definitions")
    return 1
```

### File Path Overrides

When using custom target definitions:
```python
extra_inputs = {
    "target_definition_base": Path(target_def_base),
    "target_definition_files": [
        "exoplanet",
        "auxiliary-standard",
        "monitoring-standard",
        "occultation-standard",
    ],
}
```

This tells the pipeline to generate and use fresh manifests.

## Pipeline Integration

The script leverages existing `pandorascheduler_rework.pipeline` functionality:

1. **`_generate_target_manifests()`** - Called when `target_definition_base` is set
2. **`_maybe_generate_visibility()`** - Called when `generate_visibility` is True
3. Both functions are part of `build_schedule()` workflow

## Testing

Use the normal test suite:

```bash
poetry run pytest tests/
```

### New Capability
```bash
poetry run python run_scheduler.py \
    --start "2026-02-05" --end "2026-02-12" \
    --output ./output \
    --target-definitions /path/to/PandoraTargetList/target_definition_files \
    --generate-visibility \
    --gmat-ephemeris /path/to/ephemeris.txt \
    --show-progress
```
Generates everything from definitions

## Error Handling

### Improved Error Messages

**Before:**
```
Target manifest not found: /path/to/exoplanet_targets.csv
Please ensure the legacy data directory exists.
```

**After:**
```
Target manifest not found: /path/to/exoplanet_targets.csv
Please provide target definitions via --target-definitions
or ensure manifests exist under the selected --output directory.
```

### Validation

- Checks if target definition directory exists before processing
- Warns if visibility generation requested without target definitions
- Only validates manifest existence when NOT generating from definitions

## Benefits

1. **Self-Contained Execution** - No dependency on pre-generated manifests
2. **Version Control** - Target definitions in JSON are easier to track than CSV
3. **Reproducibility** - Same definitions always produce same manifests
4. **Flexibility** - Can run with or without visibility generation
5. **Explicit Configuration** - All paths are provided via command line, no hidden defaults
6. **Testing** - Easy to test with different target definition sets

## Future Enhancements

Potential improvements for future versions:

1. **Caching** - Reuse visibility files if they exist and are up-to-date
2. **Parallel Generation** - Generate visibility for multiple targets concurrently
3. **Incremental Updates** - Only regenerate changed targets
4. **Validation** - Verify target definition integrity before processing
5. **Progress Tracking** - Show detailed progress for visibility generation phase

## Migration Guide

### For Existing Users

If you previously relied on legacy manifests under `src/pandorascheduler/data/`, migrate those manifests/visibility products into your chosen `--output` directory (for example, `output/data/`), or run with `--target-definitions` to regenerate them.

### For New Users (Recommended Workflow)

1. Clone PandoraTargetList repository
2. Run pipeline with explicit path:
   ```bash
   poetry run python run_scheduler.py \
       --start "2026-02-05" --end "2027-02-05" \
       --output ./output \
       --target-definitions /path/to/PandoraTargetList/target_definition_files \
    --generate-visibility \
    --gmat-ephemeris /path/to/ephemeris.txt \
       --show-progress
   ```
3. Subsequent runs can omit `--generate-visibility` to reuse data

### For CI/CD

```bash
# Run full pipeline with explicit paths
poetry run python run_scheduler.py \
    --start "${START_DATE}" \
    --end "${END_DATE}" \
    --output ./pipeline_output \
    --target-definitions /path/to/target_definition_files \
    --generate-visibility \
    --gmat-ephemeris /path/to/ephemeris.txt \
    --show-progress
```

## Summary

The updated `run_scheduler.py` script now provides a complete, self-contained pipeline that can:
- Generate target manifests from JSON definitions
- Create visibility catalogs from orbital ephemeris
- Run scheduling algorithms
- Produce all final outputs (schedule, XML, reports)

All with a single command, making it ideal for:
- Development and testing
- Automated pipelines
- Reproducible research
- Version-controlled target management
