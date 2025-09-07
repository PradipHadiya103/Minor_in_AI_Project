# Smoke-test: train tuned model using hyperparam_best.json and train_questions_limited.csv
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from lightgbm import LGBMClassifier
import joblib

DATA_DIR = Path(r'C:\Users\Pradip-B.Hadiya\Downloads\AIProject\dataset')
BEST_JSON = DATA_DIR / 'hyperparam_best.json'
CSV_PATH = DATA_DIR / 'train_questions_limited.csv'
MODEL_OUT = DATA_DIR / 'lgbm_tuned_model_smoke.pkl'

print('Checking files...')
if not CSV_PATH.exists():
    print('Missing CSV:', CSV_PATH)
    raise SystemExit(1)
if not BEST_JSON.exists():
    print('Missing best params JSON, continuing with defaults')
    best_params = {}
else:
    best_params = json.loads(BEST_JSON.read_text()).get('best_params', {}) or {}

print('Loading CSV...')
df = pd.read_csv(CSV_PATH)
df = df[df['target'].notna()].reset_index(drop=True)
X = df.drop(columns=['target'])
y = df['target'].astype(int)
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        X[col] = X[col].astype('category').cat.codes
X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

stratify_arg = None
if y.nunique() > 1 and (y.value_counts().min() >= 2):
    stratify_arg = y

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)

model = LGBMClassifier(random_state=42, n_jobs=-1, **best_params)
print('Model params:', model.get_params())
model.fit(X_train, y_train)

proba = model.predict_proba(X_val)[:,1]
auc = roc_auc_score(y_val, proba)
acc = accuracy_score(y_val, (proba>0.5).astype(int))
print(f'AUC={auc:.4f}, ACC={acc:.4f}')
joblib.dump(model, MODEL_OUT)
print('Saved smoke model to', MODEL_OUT)
