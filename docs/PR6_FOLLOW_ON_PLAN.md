# PR #6 Follow-On Implementation Plan

## Scope

This document captures the remaining PR #6 ideas that still look worth implementing after the minimal-fix branch.

It intentionally excludes parity-specific work and excludes large tooling additions such as the Streamlit app, notebooks, devcontainer, and `requirements.txt`.

## Already Done

Phase 0 is complete on `pr6-minimal-fixes`.

Completed work:

1. Run-scoped data directory routing across the runner, pipeline, scheduler paths, visibility output, and XML input lookup.
2. XML predefined ROI slot 0 alignment so the first ROI uses the target RA/DEC.
3. Regression coverage for both of those fixes.
4. Example config and config documentation updates.

Primary files already changed:

- `run_scheduler.py`
- `src/pandorascheduler_rework/config.py`
- `src/pandorascheduler_rework/pipeline.py`
- `src/pandorascheduler_rework/scheduler.py`
- `src/pandorascheduler_rework/observation_utils.py`
- `src/pandorascheduler_rework/visibility/catalog.py`
- `src/pandorascheduler_rework/xml/parameters.py`
- `tests/test_pipeline_visibility.py`
- `tests/test_xml.py`
- `example_scheduler_config.json`
- `docs/EXAMPLE_SCHEDULER_CONFIG.md`

## Recommended Remaining Work

Implement the remainder in four phases:

1. `primary_only_mode`
2. CLI exposure for existing occultation behavior knobs
3. Optional tolerant occultation time-limit handling
4. A focused single-visit occultation debug tool

## Phase 1: Primary-Only Mode

### Goal

Allow scheduling runs that only schedule primary science targets and do not fill gaps with auxiliary or monitoring targets.

### Why

- Useful for science-analysis mode and transit-yield studies.
- Useful for debugging schedule quality.
- Clearly missing today.

### Implementation

1. Add `primary_only_mode: bool = False` to `PandoraSchedulerConfig` in `src/pandorascheduler_rework/config.py`.
2. Parse it from JSON and CLI in `run_scheduler.py`.
3. Add a CLI flag such as `--primary-only`.
4. Gate non-primary gap filling in `src/pandorascheduler_rework/scheduler.py`.
5. Keep primary scheduling logic unchanged.
6. Keep occultation behavior inside scheduled primary visits unchanged unless a separate mode is added later.
7. Document the flag in `docs/EXAMPLE_SCHEDULER_CONFIG.md` and `example_scheduler_config.json`.

### Tests

1. Add a scheduler-focused test proving auxiliary scheduling is skipped when `primary_only_mode=True`.
2. Add a propagation test proving the flag is parsed and reaches the scheduler.
3. Add a regression test proving default behavior is unchanged when the flag is absent or `False`.

### Acceptance Criteria

- No auxiliary or monitoring gap-fill observations are scheduled when enabled.
- Primary target selection remains unchanged.
- Default runs behave exactly as they do today.

## Phase 2: Expose Existing Occultation Knobs in CLI

### Goal

Make the existing occultation strategy controls easy to use and reproducible from the command line.

### Why

- The logic already exists.
- The current ergonomics are weaker than they should be.
- This is low-risk and operationally useful.

### Implementation

1. Add explicit CLI flags in `run_scheduler.py` for:
   - `--use-target-list-for-occultations`
   - `--prioritise-occultations-by-slew`
   - `--no-break-occultation-sequences` or equivalent
2. Keep JSON support as-is, but make CLI precedence explicit.
3. Ensure the resolved values still populate `PandoraSchedulerConfig` cleanly.
4. Update docs in `docs/EXAMPLE_SCHEDULER_CONFIG.md` and optionally `README.md` and `QUICK_START.md`.

### Tests

1. Add CLI parsing or runner tests for precedence and defaults.
2. Add propagation tests for `prioritise_occultations_by_slew`.
3. Add propagation tests for `use_target_list_for_occultations`.

### Acceptance Criteria

- A user can reproduce occultation strategy from CLI alone.
- Existing JSON-driven runs continue to work unchanged.

## Phase 3: Optional Tolerant Occultation Time-Limit Handling

### Goal

Allow exploratory runs to continue when occultation manifests are incomplete, while keeping strict validation as the default.

### Why

- Improves resilience for partially curated data.
- Targets a real operational pain point without weakening the default path.

### Current Behavior

`_get_occultation_time_limit` in `src/pandorascheduler_rework/science_calendar.py` hard-fails on missing catalog data, missing target rows, or missing `Number of Hours Requested`.

### Implementation

1. Add `strict_occultation_time_limits: bool = True` to `PandoraSchedulerConfig` in `src/pandorascheduler_rework/config.py`.
2. Parse it in `run_scheduler.py`.
3. Add a CLI flag such as `--relaxed-occultation-time-limits`.
4. Update `src/pandorascheduler_rework/science_calendar.py` so that:
   - strict mode keeps today's exceptions
   - relaxed mode logs a warning and returns a very large effective limit
5. Document the tradeoff in `docs/EXAMPLE_SCHEDULER_CONFIG.md`.

### Tests

1. Keep existing strict-mode tests unchanged.
2. Add relaxed-mode tests for:
   - missing catalog
   - missing target row
   - missing `Number of Hours Requested`
3. Add one builder-level test proving XML generation continues under relaxed mode.

### Acceptance Criteria

- Default behavior remains strict.
- Relaxed mode never silently hides missing catalog information.
- XML generation can continue for exploratory runs.

## Phase 4: Single-Visit Occultation Debug Tool

### Goal

Make it easy to explain why one visit produced a particular occultation schedule.

### Why

- Targets the hardest remaining debugging surface.
- More useful than another large comparison script.
- Helps future maintenance even if no more PR #6 code is adopted.

### Implementation

1. Add a new script under `scripts/`.
   Suggested name: `scripts/debug_occultation_visit.py`
2. Inputs should include:
   - schedule CSV path
   - data dir
   - visit index or target plus time range
   - optional occultation behavior flags
3. The script should print:
   - visit metadata
   - extracted visibility samples in window
   - visibility change indices
   - derived occultation windows
   - selected candidate list source
   - excluded targets due to time limits
   - per-pass assignment outcome
   - final emitted occultation segments
4. Reuse existing logic rather than cloning it.

### Tests

1. Prefer a light unit test around any factored helper.
2. If kept script-only, validate manually against at least one known occultation case.

### Acceptance Criteria

- A single problematic visit can be explained without instrumenting production code.
- Output is detailed enough to compare segmentation and candidate selection decisions.

## Files Likely To Change

Primary implementation files:

- `run_scheduler.py`
- `src/pandorascheduler_rework/config.py`
- `src/pandorascheduler_rework/scheduler.py`
- `src/pandorascheduler_rework/science_calendar.py`

Likely test files:

- `tests/test_config_behavior.py`
- `tests/test_config_propagation.py`
- `tests/test_occultation_deprioritization.py`
- `tests/test_xml_builder.py`
- optionally a new targeted scheduler test file

Likely docs/config files:

- `docs/EXAMPLE_SCHEDULER_CONFIG.md`
- `example_scheduler_config.json`
- `README.md`
- `QUICK_START.md`

## Suggested Execution Order

1. Implement `primary_only_mode`.
2. Add CLI exposure for current occultation knobs.
3. Add relaxed occultation time-limit mode.
4. Add the debug script.
5. Run targeted tests after each phase.

## Validation Plan

For each phase:

1. Run directly affected unit tests.
2. Add one regression test for the new behavior.
3. Keep one default-path regression to ensure existing behavior did not drift.

End-of-tranche verification:

1. Run targeted config and propagation tests.
2. Run occultation-related tests.
3. Run XML-builder tests.
4. Optionally run one short real pipeline case using existing output data.

## Explicitly Out of Scope

The following PR #6 items are not recommended for implementation in this repo at this stage:

- Streamlit visibility app
- notebooks
- devcontainer
- `requirements.txt`
- broad re-porting of occultation logic that is already present locally in evolved form