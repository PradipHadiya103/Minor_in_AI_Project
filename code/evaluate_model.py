"""Evaluate saved model and plot ROC/PR curves.
Usage:
    python code/evaluate_model.py --model dataset/lgbm_tuned_model.pkl --features dataset/train_questions_limited.csv

If no args provided it uses the repository defaults.
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)


def load_model(p: Path):
    try:
        return joblib.load(p)
    except Exception:
        import lightgbm as lgb
        return lgb.Booster(model_file=str(p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dataset/lgbm_tuned_model.pkl')
    parser.add_argument('--features', type=str, default='dataset/train_questions_limited.csv')
    parser.add_argument('--save-plots', action='store_true')
    parser.add_argument('--out-dir', type=str, default='dataset', help='Directory to save plots')
    args = parser.parse_args()

    model_path = Path(args.model)
    csv_path = Path(args.features)

    if not csv_path.exists():
        raise SystemExit(f'Features CSV not found: {csv_path}')
    if not model_path.exists():
        raise SystemExit(f'Model not found: {model_path}')

    print('Loading model from', model_path)
    model = load_model(model_path)

    print('Loading features from', csv_path)
    df = pd.read_csv(csv_path)
    df = df[df['target'].notna()].reset_index(drop=True)
    X = df.drop(columns=['target'])
    y = df['target'].astype(int)

    # simple preprocessing: encode object cols
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            X[col] = X[col].astype('category').cat.codes
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')

    # Holdout split for evaluation
    from sklearn.model_selection import train_test_split
    stratify_arg = None
    if y.nunique() > 1 and (y.value_counts().min() >= 2):
        stratify_arg = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_arg)

    # Predict
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)
    elif hasattr(model, 'predict'):
        y_proba = model.predict(X_test)
        try:
            y_pred = (y_proba > 0.5).astype(int)
        except Exception:
            y_pred = model.predict(X_test)
    else:
        raise RuntimeError('Model has no predict method')

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else float('nan')
    prec, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary', zero_division=0)

    print(f'Accuracy: {acc:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, digits=4))

    # ROC & PR curves
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC curve'); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall_vals, precision_vals)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall curve')
    plt.tight_layout()

    if args.save_plots:
        out = Path(args.out_dir)
        out.mkdir(exist_ok=True, parents=True)
        # create a model-specific filename to avoid overwrites
        model_stamp = model_path.stem.replace('.', '_')
        figpath = out / f'evaluation_plots_{model_stamp}.png'
        plt.savefig(figpath)
        print('Saved plots to', figpath)
    else:
        plt.show()

    # Feature importance
    try:
        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
        else:
            import lightgbm as lgb
            booster = model if isinstance(model, lgb.Booster) else None
            if booster is not None:
                importances = pd.Series(booster.feature_importance(importance_type='gain'), index=X_test.columns).sort_values(ascending=False)
        if importances is not None:
            print('\nTop features by importance:')
            print(importances.head(20).to_string())
    except Exception as e:
        print('Could not compute feature importance:', e)


if __name__ == '__main__':
    main()
