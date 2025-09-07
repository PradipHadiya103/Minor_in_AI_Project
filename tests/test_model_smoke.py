import joblib
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path(r"C:\Users\Pradip-B.Hadiya\Downloads\AIProject\dataset")
MODEL_CANDIDATES = [
    DATA_DIR / 'lgbm_tuned_model.pkl',
    DATA_DIR / 'lgbm_tuned_model_smoke.pkl',
    DATA_DIR / 'hyperparam_best_model.pkl',
    DATA_DIR / 'lgbm_baseline.txt'
]
CSV_PATH = DATA_DIR / 'train_questions_limited.csv'


def _find_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    pytest.skip('No model artifact found in dataset/ to run smoke test')


def _load_model(path):
    # Try joblib first (sklearn/lightgbm sklearn wrapper). If fails and file is LightGBM dump (.txt), load Booster.
    try:
        return joblib.load(path)
    except Exception:
        try:
            import lightgbm as lgb
            # LightGBM native model
            return lgb.Booster(model_file=str(path))
        except Exception as e:
            pytest.skip(f'Failed to load model {path}: {e}')


def _prepare_features(csv_path, n=5):
    if not csv_path.exists():
        pytest.skip('Features CSV not found; run feature extraction first')
    df = pd.read_csv(csv_path)
    df = df[df['target'].notna()].reset_index(drop=True)
    if df.shape[0] == 0:
        pytest.skip('No rows with target in features CSV')
    X = df.drop(columns=['target']).copy()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype('category').cat.codes
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    return X.iloc[:n]


def test_model_predict_shape():
    model_path = _find_model()
    model = _load_model(model_path)
    X = _prepare_features(CSV_PATH, n=7)
    # Run prediction depending on model type
    if hasattr(model, 'predict_proba'):
        preds = model.predict_proba(X)  # sklearn API
        # Expect shape (n,2) or (n,k)
        assert preds.shape[0] == X.shape[0]
    else:
        # Assume lightgbm.Booster
        if hasattr(model, 'predict'):
            preds = model.predict(X)
            # Booster.predict returns (n,) of probabilities
            assert preds.shape[0] == X.shape[0]
        else:
            pytest.skip('Loaded model has no predict method')


def test_feature_schema():
    """Validate the feature CSV contains expected columns and numeric types for modeling."""
    expected_cols = [
        'user_id', 'content_id', 'prior_question_elapsed_time',
        'user_prior_correct_rate', 'content_prior_correct_rate',
        'tags_count', 'is_question', 'target'
    ]
    if not CSV_PATH.exists():
        pytest.skip('Features CSV not found; run feature extraction first')
    df = pd.read_csv(CSV_PATH)
    # Ensure expected columns exist
    for c in expected_cols:
        assert c in df.columns, f'Missing expected column: {c}'
    # Quick dtype checks: numeric-like columns should be coercible to numeric
    numeric_cols = ['prior_question_elapsed_time', 'user_prior_correct_rate', 'content_prior_correct_rate', 'tags_count']
    for c in numeric_cols:
        # attempt conversion
        series = pd.to_numeric(df[c], errors='coerce')
        # at least half of values should be non-NA after coercion (tolerant check)
        non_na_fraction = series.notna().sum() / max(1, len(series))
        assert non_na_fraction >= 0.5, f'Column {c} seems mostly non-numeric'


def test_model_accuracy_sanity():
    """Simple sanity check: model AUC should be better than random (>= 0.5) on a small holdout slice."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    model_path = _find_model()
    model = _load_model(model_path)
    # prepare a slightly larger sample for a stable AUC
    if not CSV_PATH.exists():
        pytest.skip('Features CSV not found; run feature extraction first')
    df = pd.read_csv(CSV_PATH)
    df = df[df['target'].notna()].reset_index(drop=True)
    if df.shape[0] < 50:
        pytest.skip('Not enough rows to run accuracy sanity check')
    # sample 200 (or fewer) rows for quick check
    sample_n = min(200, df.shape[0])
    sample = df.sample(sample_n, random_state=1).reset_index(drop=True)
    X = sample.drop(columns=['target'])
    y = sample['target'].astype(int)
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype('category').cat.codes
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

    # get probabilities
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)

    # if all labels are the same in sample, skip AUC check
    if y.nunique() < 2:
        pytest.skip('Sampled labels are constant; cannot compute AUC')

    auc = roc_auc_score(y, probs)
    assert auc >= 0.5, f'AUC too low ({auc:.4f}); sanity check failed.'
