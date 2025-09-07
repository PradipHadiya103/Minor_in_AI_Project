"""Hyperparameter tuning for LightGBM baseline
- Loads `train_questions_limited.csv` produced by the notebook
- Performs simple preprocessing (category codes + numeric coercion)
- Samples up to SAMPLE_SIZE rows for faster tuning
- Runs RandomizedSearchCV over LightGBM classifier
- Saves best params, cv results CSV, and best model (joblib)
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score
from lightgbm import LGBMClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\Pradip-B.Hadiya\Downloads\AIProject\dataset')
CSV_PATH = DATA_DIR / 'train_questions_limited.csv'
REPORT_CSV = DATA_DIR / 'hyperparam_cv_results.csv'
BEST_JSON = DATA_DIR / 'hyperparam_best.json'
MODEL_PKL = DATA_DIR / 'hyperparam_best_model.pkl'

SAMPLE_SIZE = 40000  # sample for tuning to keep runtime modest
N_ITER = 20
CV_FOLDS = 3
RANDOM_STATE = 42

print('Loading', CSV_PATH)
if not CSV_PATH.exists():
    raise SystemExit(f'CSV not found: {CSV_PATH} - run feature extraction first')

df = pd.read_csv(CSV_PATH)
print('Rows available:', len(df))

# Drop rows missing target
df = df[df['target'].notna()].reset_index(drop=True)
if df.empty:
    raise SystemExit('No rows with target available')

# Sample for tuning
if len(df) > SAMPLE_SIZE:
    df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
    print('Sampled rows for tuning:', len(df))

# Prepare X, y
X = df.drop(columns=['target'])
y = df['target'].astype(int)

# Encode object/categorical columns
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        X[col] = X[col].astype('category').cat.codes

# Coerce to numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

# Define estimator and parameter distributions
est = LGBMClassifier(objective='binary', random_state=RANDOM_STATE, n_jobs=-1)

param_dist = {
    'n_estimators': [100, 200, 400],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [15, 31, 63, 127],
    'min_child_samples': [5, 10, 20, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.0, 0.1, 0.5],
    'reg_lambda': [0.0, 0.1, 0.5]
}

scorer = make_scorer(roc_auc_score, needs_proba=True)
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print('Starting RandomizedSearchCV')
search = RandomizedSearchCV(
    estimator=est,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring=scorer,
    n_jobs=-1,
    cv=cv,
    random_state=RANDOM_STATE,
    verbose=2,
    return_train_score=True
)

search.fit(X, y)

print('Best score (cv):', search.best_score_)
print('Best params:')
print(search.best_params_)

# Save CV results
cv_res = pd.DataFrame(search.cv_results_)
cv_res.to_csv(REPORT_CSV, index=False)
print('Saved CV results to', REPORT_CSV)

# Save best params
with open(BEST_JSON, 'w') as f:
    json.dump({'best_score': float(search.best_score_), 'best_params': search.best_params_}, f, indent=2)
print('Saved best params to', BEST_JSON)

# Refit the best estimator on full sampled data and save model
best_model = search.best_estimator_
best_model.fit(X, y)
joblib.dump(best_model, MODEL_PKL)
print('Saved best model to', MODEL_PKL)

print('Done')
