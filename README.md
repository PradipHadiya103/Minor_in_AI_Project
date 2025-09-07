# AIProject — Small student-answer prediction prototype

This repository contains a small end-to-end prototype to predict whether a student answers a question correctly. The workflow covers data loading, memory-friendly dtype casting, feature engineering (leakage-safe user/content statistics), a baseline LightGBM model, hyperparameter tuning, a reproducible tuned training step, evaluation, and tests.

This README explains the files, how to reproduce the work, how to run training/tuning/evaluation, and how to run the included tests.

---

## Recommended repository layout

Keep the repository root like this (the existing repo mostly matches this):

- `code/` — helper scripts and notebooks
  - `code/code.ipynb` — main notebook (load, feature engineering, baseline training, evaluation)
   - `code/code.ipynb` — main notebook (load, feature engineering, baseline training). Evaluation has been delegated to `code/evaluate_model.py` for reuse and clearer scripting.
   - `code/evaluate_model.py` — standalone evaluation + plotting script (ROC/PR, metrics, feature importance)
  - `code/train_baseline.py` — standalone baseline trainer (used for smoke testing)
  - `code/hyperparameter_tune.py` — RandomizedSearchCV tuner (produces `hyperparam_*` outputs)
  - `code/train_tuned_smoke.py` — smoke test for the tuned training cell
- `dataset/` — data and derived artifacts (store or symlink large files as needed)
  - `train.csv`, `questions.csv`, `lectures.csv` — raw inputs (not checked-in if large)
  - `train_questions_limited.csv` — features CSV produced by the notebook
  - `train_features_limited.parquet` or CSV fallback
  - `lgbm_tuned_model.pkl`, `lgbm_tuned_model_smoke.pkl`, `lgbm_baseline.txt` — saved models
  - `hyperparam_cv_results.csv`, `hyperparam_best.json`, `hyperparam_best_model.pkl` — tuning artifacts
- `tests/` — pytest tests
  - `tests/test_model_smoke.py` — smoke tests (model load + prediction shape and small accuracy/schema checks)
- `README.md` — this file
- `requirements.txt` — Python dependencies
- `.gitignore` — recommended ignores


## Quick setup (Windows PowerShell)

1. Create and activate a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Make sure `dataset/` contains the raw CSVs (`train.csv`, `questions.csv`, `lectures.csv`) or place the preprocessed CSV `train_questions_limited.csv` there if you want to skip feature extraction.


## Reproduce the pipeline (high-level)

1. Feature extraction (notebook):
   - Open `code/code.ipynb` and run the cells in order to load the raw CSVs, compute leakage-safe cumulative features, and save the `train_questions_limited.csv` feature file to `dataset/`.
   - The notebook limits `train.csv` to the first 100,000 rows by default for quick prototyping.

2. Baseline training (quick):
   - Run the standalone baseline trainer (mirrors the notebook training cell):

```powershell
python "code/train_baseline.py"
```

   - This trains a simple LightGBM baseline using the saved `train_questions_limited.csv` and writes model & metrics to `dataset/`.

3. Hyperparameter tuning (optional, longer):
   - Run the tuner (samples up to 40k rows by default; adjust `SAMPLE_SIZE` in the script):

```powershell
python "code/hyperparameter_tune.py"
```

   - This runs `RandomizedSearchCV`, saves `hyperparam_cv_results.csv`, `hyperparam_best.json`, and a pickled best model to `dataset/`.

4. Reproducible tuned training:
   - Use the notebook cell or run the smoke script:

```powershell
python "code/train_tuned_smoke.py"
```

   - The notebook's reproducible training cell reads `hyperparam_best.json` for params, trains a final `LGBMClassifier`, evaluates on a holdout split, and saves `lgbm_tuned_model.pkl` and `lgbm_tuned_metrics.txt` in `dataset/`.

5. Evaluation & plots
   - Run the standalone evaluation script to compute Accuracy, AUC, Precision/Recall/F1, confusion matrix, and plot ROC / PR curves. Example (prints metrics only):

```powershell
python "code/evaluate_model.py" --model "dataset/lgbm_tuned_model.pkl" --features "dataset/train_questions_limited.csv"
```

   - To save the ROC/PR plots to disk use `--save-plots` and control the output directory with `--out-dir`. The script will write a model-specific file named `evaluation_plots_{model_stem}.png` into the output directory (for example `dataset/evaluation_plots_hyperparam_best_model.png`). Example:

```powershell
python "code/evaluate_model.py" --model "dataset/hyperparam_best_model.pkl" --features "dataset/train_questions_limited.csv" --save-plots --out-dir dataset
```

   - Alternatively, the notebook's final cell now calls this script so opening `code/code.ipynb` and running the last cell will execute the same script.

6. Tests
   - Run pytest to validate the saved model and basic schema:

```powershell
python -m pytest -q
```


## Files you will see produced

- `dataset/train_questions_limited.csv` — features for modeling.
- `dataset/train_features_limited.parquet` or `train_features_limited.csv` — full features saved.
- `dataset/lgbm_baseline.txt` — LightGBM native baseline (if produced).
- `dataset/lgbm_tuned_model.pkl` — final tuned model (sklearn wrapper / joblib dump).
- `dataset/lgbm_tuned_model_smoke.pkl` — smoke training output.
- `dataset/hyperparam_cv_results.csv`, `dataset/hyperparam_best.json`, `dataset/hyperparam_best_model.pkl` — tuning artifacts.
- `dataset/lgbm_metrics.txt`, `dataset/lgbm_tuned_metrics.txt` — human-readable metrics.


## Notes and recommendations

- Large raw CSVs: if `train.csv` is very large, avoid committing it to Git. Instead either:
  - Commit a small sample for testing under `dataset/sample/` and document it in README; or
  - Keep raw data outside git and provide a small script to download / prepare it.

- Reproducibility:
  - Use `MAX_TRAIN_ROWS` in the notebook to control the prototyping sample.
  - Ensure `random_state` seeds are set (scripts already include `random_state=42`).

- Improving evaluation realism:
  - Use a time-based or user-based holdout to avoid leakage rather than a random split if your events are sequential.

- CI recommendation:
  - Add a GitHub Actions workflow that runs `pip install -r requirements.txt` and `pytest` on push. I can add a starter workflow if you want.


## Contact / next steps I can do for you

- Add a minimal `.github/workflows/ci.yml` to run tests.
- Create a small `dataset/sample/` folder with a tiny sample CSV (if you want me to create one from current data for quick uploads).
- Tweak `requirements.txt` to pin specific versions.

If you want, I can now create the CI workflow and/or the small data sample and then run the tests in CI locally. Let me know which one to do next.
