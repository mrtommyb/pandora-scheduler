# Pandora Scheduler (rework)

Brief overview and quick links for developers working on the rework.

- See `QUICK_START.md` for runnable examples and common workflows.
- Example JSON configuration: `example_scheduler_config.json` (root) — use with `--config`.
- Detailed keys: `docs/EXAMPLE_SCHEDULER_CONFIG.md`.

## Quick Start

```bash
# 1) Install dependencies (poetry environment assumed)
poetry install

# 2) Run a quick test (assumes manifests/visibility already present)
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-07" \
    --output ./output_test

# 3) Full pipeline with target definitions
poetry run python run_scheduler.py \
    --start "2026-02-05" \
    --end "2026-02-12" \
    --output ./output \
    --target-definitions /path/to/PandoraTargetList/target_definition_files \
    --generate-visibility \
    --gmat-ephemeris /path/to/ephemeris.txt \
    --show-progress
```

If you need help, read `QUICK_START.md` for examples and troubleshooting tips.

## Visible Dayside App

The repo includes a Streamlit visualization app for inspecting Sun/Moon/Earth
keepout geometry:

```bash
pip install -r vis_app/pandora_visible_dayside_requirements.txt
streamlit run vis_app/pandora_visible_dayside_app.py
```

Deployment notes:

- App entrypoint: `vis_app/pandora_visible_dayside_app.py`
- Requirements: `vis_app/pandora_visible_dayside_requirements.txt`
- Deployment guide: `vis_app/DEPLOY_pandora_visible_dayside.md`
