# AML Cohort Explorer

Individual Streamlit adaptation of Martin d’Halloy's ENS survival data challenge project.
Explore clinical measurements, gene prevalence and right-censored follow-up, then
evaluate a small Cox survival baseline. This is an educational research application,
not a clinical decision tool.

## Quick start (Python 3.12)

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open http://localhost:8501. No account, API key or model download is required.
The default **Synthetic demo** contains 160 fictional patients and fictional genes.

## Use the app

1. Choose **Synthetic demo**, **Repository data**, or **Upload CSV files**.
2. Select centers and the bone-marrow-blast range. Bounds are inclusive.
   An empty center selection selects no patients. Missing blast values can be included.
3. Explore clinical distributions and missingness.
4. Inspect mutation prevalence (distinct patients per gene / all selected patients).
5. View a Kaplan–Meier curve using patients with follow-up; censored observations are handled.
6. Click **Train and evaluate baseline** to train once and display held-out scores.
7. Export the filtered clinical table or the held-out predictions.

The baseline is intentionally small: six numeric clinical measurements, median
imputation, standardization, and ridge-regularized Cox PH (alpha=1).
The split is 75% training / 25% test, stratified by event status, seed 42.
Imputation and scaling fit only on the training partition. At least 40 labelled
patients and 8 per status are required. The UI reports one held-out **Harrell C-index**.
Repeatedly selecting cohorts based on that score is not independent validation.
Clinical assumptions, subgroup stability and external validity have not been assessed.

Missing measurements are allowed, invalid numeric strings/infinities/negative values
are rejected. Clinical/target IDs must be unique and are preserved as strings.
Molecular and target IDs absent from the clinical table are rejected.
Patients without mutation rows are retained in prevalence denominators; absence of a
record is not proof of a negative assay.
Repository follow-up rows with missing status/time are excluded with an explicit count.
Uploaded target files must have complete follow-up rows; follow-up is optional.

## CSV contracts

| File | Required columns | Rules |
| --- | --- | --- |
| Clinical | ID, CENTER, BM_BLAST, WBC, ANC, MONOCYTES, HB, PLT | Unique nonempty ID; numeric measurements may be missing; BM_BLAST 0–100 |
| Molecular (optional on upload) | ID, GENE | Multiple mutation rows per patient allowed |
| Target (optional on upload) | ID, OS_YEARS, OS_STATUS | Unique ID; finite nonnegative years; status 0=censored, 1=event |

Use comma-separated UTF-8 CSVs. Demo examples are in `demo/`.
The app does not persist uploaded files to disk.

## Run with Docker

Install and start Docker Desktop on macOS, then:

```bash
docker build -t aml-explorer .
docker run --rm -p 127.0.0.1:8501:8501 aml-explorer
```

Open http://localhost:8501. The image runs as a non-root user. It includes
**only code, dependencies and synthetic demo CSVs**, never `data/` or `outputs/`.
The image health check probes `/_stcore/health`.

To use your local challenge files without adding them to the image:

```bash
docker run --rm -p 127.0.0.1:8501:8501 -v "$(pwd)/data:/app/data:ro" aml-explorer
```

Then choose **Repository data**. Without that mount, choose demo or upload mode.

## Tests and CI

```bash
python -m pytest -q
```

Tests cover malformed CSVs, schemas, duplicates, ID preservation, numeric validity,
inclusive filters, empty results, missing follow-up, gene-count denominators,
censoring/tied times, model reproducibility and Streamlit interactions.
Tests use synthetic data only.

`.github/workflows/ci.yml` runs on pushes and pull requests:
1. Install the pinned environment and run pytest.
2. Build the Docker image.
3. Execute a Streamlit AppTest inside the image.
4. Start the container, verify HTTP health and confirm that real-data/model directories
   are absent.

GitHub Actions must be enabled, with available runner capacity. Check the run status
before claiming the Docker build passed. The workflow does not publish an image.

## Reproducibility

- Python 3.12; Docker base fixes the Python patch and Debian variant.
- `requirements.in` lists direct dependencies; `requirements.txt` pins the full tested environment.
- Recreate the environment with `pip install -r requirements.txt`.
- Seed 42 controls both the demo generator and train/test split.
- Regenerate examples with `python demo/generate.py`.
- `demo/SHA256SUMS` records the demo files used for this version.
- No network requests or pretrained-model downloads are required at app runtime.
- Docker base tags can be updated upstream; for bit-for-bit archival, record the
  built image digest and platform. Numerical libraries/platforms may produce small differences.

## Relationship to the original project

`train_and_submit.py` is the original competition training script, retained unchanged.
It builds clinical/molecular features and combines Coxnet with gradient boosting.
Its saved `outputs/model.joblib` is not loaded by the app: its original environment
and compatibility have not been verified. The original ensemble's rank-based scores
depend on the prediction cohort and are not survival probabilities.

The app's lightweight clinical baseline is a new reproducible demonstration, not a
reproduction of competition performance. Its Harrell metric differs from the original
IPCW C-index at tau=7. No leaderboard score is claimed for this baseline.

The original script remains available for further research; a complete historical
training reproduction is outside the validated app workflow. Its learned gene/category
selection occurs before cross-validation and should be moved inside folds before
using those CV results as an unbiased performance estimate.

## Data provenance and submission

The existing `data/` files come from the original ENS challenge work. Their redistribution
terms have not been established here. The repository remains private; do not assume
that existing Git tracking authorizes public redistribution.
Use synthetic data for any public container demonstration. Contact the instructors
via Slack if confidentiality prevents making the submission repository public, as
specified in the assignment.

The original project is credited to its existing Git history. This Streamlit adaptation
was developed with AI assistance and should be reviewed, understood and disclosed
according to course rules.

Submit this repository's URL after reviewing/merging the adaptation branch.
Publishing an image to Docker Hub is optional and has not been performed.

## Layout

- `streamlit_app.py`: interface.
- `aml/data.py`: importing, validation, filtering, prevalence.
- `aml/model.py`: Kaplan–Meier and lightweight Cox evaluation.
- `tests/`: unit and app interaction tests.
- `demo/`: fictional reproducible fixtures.
- `Dockerfile`, `.dockerignore`: container packaging.
- `.github/workflows/ci.yml`: automated verification.
- `train_and_submit.py`, `data/`, `outputs/`: original research project.
