# Standalone LightGBM baseline trainer
# This script mirrors the notebook training cell but runs as a plain Python script.

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

DATA_DIR = Path(r'C:\Users\Pradip-B.Hadiya\Downloads\AIProject\dataset')
CSV_PATH = DATA_DIR / 'train_questions_limited.csv'
MODEL_PATH = DATA_DIR / 'lgbm_baseline.txt'

print('Loading', CSV_PATH)
if not CSV_PATH.exists():
    print('ERROR: CSV not found:', CSV_PATH)
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print('Initial rows:', len(df))

# Drop rows without target
df = df[df['target'].notna()].reset_index(drop=True)
if df.empty:
    print('No training rows with target found. Run feature extraction first.')
    sys.exit(1)

# Prepare features and target
X = df.drop(columns=['target'])
y = df['target'].astype(int)

# Encode object/categorical columns
for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        X[col] = X[col].astype('category').cat.codes

# Coerce to numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

# Safe stratify
stratify_arg = None
if y.nunique() > 1 and (y.value_counts().min() >= 2):
    stratify_arg = y

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)

train_data = lgb.Dataset(X_train.values, label=y_train.values)
valid_data = lgb.Dataset(X_val.values, label=y_val.values, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'seed': 42
}

print('Starting training...')
bst = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    callbacks=[lgb.callback.early_stopping(20), lgb.callback.log_evaluation(50)],
)

best_it = getattr(bst, 'best_iteration', None)
if best_it:
    y_pred_proba = bst.predict(X_val.values, num_iteration=best_it)
else:
    y_pred_proba = bst.predict(X_val.values)

auc = roc_auc_score(y_val, y_pred_proba)
acc = accuracy_score(y_val, (y_pred_proba > 0.5).astype(int))
print(f'Validation AUC: {auc:.4f}, Accuracy: {acc:.4f}')

# Feature importance
fi = pd.DataFrame({'feature': X.columns, 'importance': bst.feature_importance()}).sort_values('importance', ascending=False)
print('\nTop features:')
print(fi.head(20).to_string(index=False))

# Save model and metrics
bst.save_model(str(MODEL_PATH))
print('Saved model to', MODEL_PATH)
metrics_path = DATA_DIR / 'lgbm_metrics.txt'
with open(metrics_path, 'w') as f:
    f.write(f'Validation AUC: {auc:.4f}\n')
    f.write(f'Validation Accuracy: {acc:.4f}\n')
print('Saved metrics to', metrics_path)
