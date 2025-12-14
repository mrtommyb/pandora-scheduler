**Example Scheduler JSON Configuration**

**Purpose**: This document explains the keys accepted by the scheduler JSON configuration file (passed to `run_scheduler.py --config <file>`). Use `example_scheduler_config.json` at the repository root as a ready-made template.

How to use
- Create or edit a JSON file (for example, `my_sched_config.json`) and pass it to the runner:

```fish
poetry run python run_scheduler.py --start "2026-02-05" --end "2026-02-12" \
  --output ./output --config my_sched_config.json --target-definitions /path/to/PandoraTargetList/target_definition_files
```

CLI arguments take precedence over JSON values when both are provided.

Scope
- The keys below correspond to the fields on `PandoraSchedulerConfig` (see `src/pandorascheduler_rework/config.py`). Default values shown are the library defaults used when keys are omitted.

Timing & window
- `schedule_step_hours` (float, default `24.0`): scheduler rolling window step size in hours.
- `commissioning_days` (int, default `0`): number of commissioning days at mission start.

Paths & data sources
- `extra_inputs.target_definition_base` (string): path to PandoraTargetList target definition files (example: `/path/to/PandoraTargetList/target_definition_files`).
- `extra_inputs.visibility_gmat` (string): path to GMAT ephemeris file used to generate visibilities (can also be provided via CLI `--gmat-ephemeris`).

Scheduling thresholds
- `transit_coverage_min` (float 0-1, default `0.2`): minimum transit coverage to consider scheduling.
- `min_visibility` (float, default `0.0`): minimum visibility fraction for considering a window.

Auxiliary requested-time behavior
- Auxiliary/standard targets are capped using the per-target `Number of Hours Requested` values from the target manifests (rather than a global config threshold).
- If no eligible non-primary targets are visible in a window, the scheduler will fall back to scheduling a target that has already met its requested hours and will emit warnings when this occurs.
- The Observation Time Report uses the same `Number of Hours Requested` values to compute requested-vs-scheduled deltas for non-primary targets.

Weighting factors
- `transit_scheduling_weights` (array of 3 floats, default `[0.8, 0.0, 0.2]`): tuple representing (coverage, saa, schedule) weights. Must sum to 1.0.

Keepout / avoidance angles (degrees)
- `visibility_sun_deg` / `sun_avoidance_deg` (float, default `91.0`)
- `visibility_moon_deg` / `moon_avoidance_deg` (float, default `25.0`)
- `visibility_earth_deg` / `earth_avoidance_deg` (float, default `86.0`)

XML generation parameters
- `obs_sequence_duration_min` (int, default `90`): default observation sequence length used when writing the science calendar XML.
- `occ_sequence_limit_min` (int, default `50`): maximum occultation sequence length in minutes for XML emission.
- `min_sequence_minutes` (int, default `5`): minimum sequence length to include in XML output.
- `break_occultation_sequences` (bool, default `true`): whether to break long occultation sequences into chunks.

Standard star observations
- `std_obs_duration_hours` (float, default `0.5`)
- `std_obs_frequency_days` (float, default `3.0`)
- If no standard star is fully visible during the standard-star observation window, the scheduler logs a warning.

Behavior flags
- `show_progress` (bool, default `false`): show progress bars during processing.
- `force_regenerate` (bool, default `false`): force regeneration of intermediate files even if they already exist.
- `use_target_list_for_occultations` (bool, default `false`): use the target list for occultation scheduling instead of a separate list.
- `prioritise_occultations_by_slew` (bool, default `false`): prioritise occultation targets based on slew cost.

Auxiliary sorting & metadata
- `aux_sort_key` (string, default `"sort_by_tdf_priority"`): key used to sort auxiliary targets
- `author` (string): author metadata to add to generated XML
- `created_timestamp` (string or datetime): creation timestamp to add to XML metadata
- `visit_limit` (int or null): optionally limit the total number of visits (useful for tests)
- `target_filters` (array of strings): filters applied when generating visibility catalogs

Extra inputs (pipeline-specific)
- `extra_inputs` (object): container for additional path overrides consumed by pipeline stages. Common keys include:
  - `target_definition_base`: path to PandoraTargetList files
  - `target_definition_files`: list of which categories to convert into manifests (e.g. `["exoplanet","auxiliary-standard","monitoring-standard","occultation-standard"]`)
  - `generate_visibility`: boolean-like value to request visibility generation
  - `visibility_gmat`: path to GMAT ephemeris file
  - `visibility_output_root`: optional override for where visibility files are written
  - `skip_manifests`: if true, skip regenerating target manifests (useful during iterative profiling)

Notes & tips
- The example file `example_scheduler_config.json` in the repository root contains the keys above and can be used as a starting point.
- Most keys may be provided in the JSON config; a subset of common options are still available on the CLI (weights, keepout angles, `--skip-manifests`, etc.). CLI flags take precedence.
- If you need additional keys added to the `PandoraSchedulerConfig`, update `src/pandorascheduler_rework/config.py` and ensure `create_scheduler_config` (in `run_scheduler.py`) maps them through.

If you'd like, I can copy this content into `README.md` or expand it into a short examples page under `docs/`.
