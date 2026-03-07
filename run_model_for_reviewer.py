#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
from scipy.signal import savgol_filter

def add_window_features(df, windows):
    X = df.copy()
    for (a,b) in windows:
        cols = [f'v{i}' for i in range(a, b+1) if f'v{i}' in X.columns]
        if not cols: continue
        X[f'int_{a}_{b}'] = X[cols].sum(axis=1)
        X[f'max_{a}_{b}'] = X[cols].max(axis=1)
        X[f'mean_{a}_{b}'] = X[cols].mean(axis=1)
    return X

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default="label")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    feat_cols = [c for c in df.columns if c.startswith('v')]
    X = df[feat_cols].copy()
    y = df[args.target].copy()
    try:
        X[feat_cols] = X[feat_cols].apply(lambda r: savgol_filter(r.values, window_length=11, polyorder=3), axis=1, result_type='expand')
    except Exception:
        pass
    windows = [(450,500),(315,330)]
    X = add_window_features(X, windows)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, stratify=y_train, random_state=42)
    iterations = 200 if args.quick else 800
    from collections import Counter
    counts = Counter(y_train); total = sum(counts.values())
    classes = sorted(counts.keys())
    class_weights = [total/counts[c] for c in classes]
    model = CatBoostClassifier(iterations=iterations, learning_rate=0.03, depth=6, class_weights=class_weights, random_seed=42, verbose=100, use_best_model=True)
    model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val), early_stopping_rounds=50)
    model.save_model(str(Path(args.out_dir)/"catboost_model.cbm"))
    joblib.dump(LabelEncoder().fit(y_train), Path(args.out_dir)/"label_encoder.joblib")
    preds = model.predict(X_test)
    (Path(args.out_dir)/"classification_report_test.txt").write_text(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    (Path(args.out_dir)/"confusion_matrix_test.txt").write_text(str(cm))
    fi = model.get_feature_importance(prettified=False)
    feat_names = X_train.columns.tolist()
    pd.DataFrame({"feature":feat_names,"importance":fi}).sort_values("importance",ascending=False).to_csv(Path(args.out_dir)/"feature_importance.csv", index=False)
    np.save(Path(args.out_dir)/"best_thresholds.npy", np.ones(len(fi))*0.5)
    print("Artifacts saved to", args.out_dir)

if __name__ == "__main__":
    main()
