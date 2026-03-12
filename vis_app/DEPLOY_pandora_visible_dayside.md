# Pandora Visible Dayside Deployment

## Local run

From the repo root:

```bash
pip install -r vis_app/pandora_visible_dayside_requirements.txt
streamlit run vis_app/pandora_visible_dayside_app.py
```

The app uses the repo-local lookup file:

```text
vis_app/exoplanet_target_lookup.csv
```

## Streamlit Community Cloud

Use these settings:

- Repository: this repo
- Branch: the branch containing the app
- Main file path: `vis_app/pandora_visible_dayside_app.py`

Dependencies:

- If the deployment UI asks for a requirements file, use:
  `vis_app/pandora_visible_dayside_requirements.txt`

## Notes

- The Simbad free-text resolver requires `astroquery`.
- If internet access is unavailable in the deployment environment, the
  exoplanet dropdown still works because it is backed by the static CSV
  shipped in the repo.
