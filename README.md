# Dashboard

This folder contains the Streamlit application used to explore the SAPCS cohort and the model outputs.

## What you need first

- Python 3.10 or newer
- The dashboard dependencies from `dashboard/requirements.txt`
- A processed cohort CSV named `merged_feature_matrix_SA.csv`

## Data file expected by the app

The app looks for the processed file at:

```text
dashboard/data/processed/merged_feature_matrix_SA.csv
```

If the file is missing, the app stops and explains what to do.

To create it:

1. Place `41586_2022_5154_MOESM3_ESM.xlsx` in `data/raw`.
2. Open `notebooks/12_Summary_and_Feature_Matrix.ipynb`.
3. Run the notebook to generate `data/processed/merged_feature_matrix_SA.csv`.
4. Copy that CSV into `dashboard/data/processed/merged_feature_matrix_SA.csv`.

## Local setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already ran `make requirements`, the dashboard dependencies are already installed.

## Run the app

From the repository root:

```bash
streamlit run dashboard/app.py
```

Or, if you prefer the Makefile wrapper:

```bash
make dashboard-run
```

To use a different port:

```bash
streamlit run dashboard/app.py --server.port 8502
```

## Notes for users

- Global filters in the sidebar update all tabs.
- The model tab uses the processed cohort file and does not depend on the sidebar filters.
- If the app cannot find data, check both `data/processed` and `dashboard/data/processed`.
